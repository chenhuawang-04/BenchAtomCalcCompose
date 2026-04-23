#include "benchcalc/parser.h"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace benchcalc {
namespace {

enum class LexKind {
    Identifier,
    Number,
    Operator,
    LParen,
    RParen,
};

struct LexToken {
    LexKind kind{};
    std::string text;
    std::size_t pos = 0;
};

bool is_ident_start(char c) {
    return std::isalpha(static_cast<unsigned char>(c)) != 0 || c == '_';
}

bool is_ident_part(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
}

bool is_digit(char c) {
    return std::isdigit(static_cast<unsigned char>(c)) != 0;
}

bool is_number_start(char c) {
    return is_digit(c) || c == '.';
}

bool parse_number_lexeme(std::string_view expr, std::size_t start, std::size_t& end_out) {
    std::size_t i = start;
    bool has_digit = false;

    while (i < expr.size() && is_digit(expr[i])) {
        has_digit = true;
        ++i;
    }

    if (i < expr.size() && expr[i] == '.') {
        ++i;
        while (i < expr.size() && is_digit(expr[i])) {
            has_digit = true;
            ++i;
        }
    }

    if (!has_digit) {
        return false;
    }

    if (i < expr.size() && (expr[i] == 'e' || expr[i] == 'E')) {
        const std::size_t e_pos = i;
        ++i;

        if (i < expr.size() && (expr[i] == '+' || expr[i] == '-')) {
            ++i;
        }

        const std::size_t exp_begin = i;
        while (i < expr.size() && is_digit(expr[i])) {
            ++i;
        }

        // e 后无数字，回退
        if (i == exp_begin) {
            i = e_pos;
        }
    }

    end_out = i;
    return true;
}

std::vector<LexToken> tokenize(std::string_view expr) {
    std::vector<LexToken> out;
    std::size_t i = 0;

    while (i < expr.size()) {
        const char c = expr[i];

        if (std::isspace(static_cast<unsigned char>(c)) != 0) {
            ++i;
            continue;
        }

        if (is_ident_start(c)) {
            const std::size_t start = i;
            ++i;
            while (i < expr.size() && is_ident_part(expr[i])) {
                ++i;
            }
            out.push_back(LexToken{LexKind::Identifier, std::string(expr.substr(start, i - start)), start});
            continue;
        }

        if (is_number_start(c)) {
            const std::size_t start = i;
            std::size_t end = i;
            if (!parse_number_lexeme(expr, start, end)) {
                throw std::runtime_error("Invalid number at position " + std::to_string(start));
            }
            i = end;
            out.push_back(LexToken{LexKind::Number, std::string(expr.substr(start, end - start)), start});
            continue;
        }

        if (c == '+' || c == '-' || c == '*' || c == '/') {
            out.push_back(LexToken{LexKind::Operator, std::string(1, c), i});
            ++i;
            continue;
        }

        if (c == '(') {
            out.push_back(LexToken{LexKind::LParen, "(", i});
            ++i;
            continue;
        }

        if (c == ')') {
            out.push_back(LexToken{LexKind::RParen, ")", i});
            ++i;
            continue;
        }

        throw std::runtime_error("Unexpected character at position " + std::to_string(i) + ": '" + c + "'");
    }

    return out;
}

int precedence(OpCode op) {
    switch (op) {
    case OpCode::Add:
    case OpCode::Sub:
        return 1;
    case OpCode::Mul:
    case OpCode::Div:
        return 2;
    case OpCode::Sin:
    case OpCode::Sqrt:
        return 3;
    }
    return 0;
}

OpCode binary_from_char(char c, std::size_t pos) {
    switch (c) {
    case '+': return OpCode::Add;
    case '-': return OpCode::Sub;
    case '*': return OpCode::Mul;
    case '/': return OpCode::Div;
    default:
        throw std::runtime_error("Unknown binary operator at position " + std::to_string(pos));
    }
}

OpCode function_from_name(std::string_view name, std::size_t pos) {
    if (name == "sin") {
        return OpCode::Sin;
    }
    if (name == "sqrt") {
        return OpCode::Sqrt;
    }

    throw std::runtime_error("Unknown function at position " + std::to_string(pos) + ": " + std::string(name));
}

struct StackToken {
    enum class Kind {
        LParen,
        BinaryOp,
        Function,
    };

    Kind kind{};
    OpCode op = OpCode::Add;
    std::size_t pos = 0;
};

} // namespace

ParsedExpression parse_expression(std::string_view expression) {
    auto lex = tokenize(expression);

    if (lex.empty()) {
        throw std::runtime_error("Expression is empty");
    }

    ParsedExpression parsed;
    parsed.original = std::string(expression);

    std::unordered_map<std::string, std::uint16_t> var_to_index;
    std::vector<StackToken> op_stack;

    auto push_variable = [&](const std::string& name) {
        auto it = var_to_index.find(name);
        std::uint16_t idx = 0;

        if (it == var_to_index.end()) {
            idx = static_cast<std::uint16_t>(parsed.variables.size());
            parsed.variables.push_back(name);
            var_to_index.emplace(name, idx);
        } else {
            idx = it->second;
        }

        parsed.rpn.push_back(RpnToken{RpnToken::Kind::Variable, idx, OpCode::Add});
    };

    auto push_constant = [&](const std::string& text, std::size_t pos) {
        char* end = nullptr;
        const float value = std::strtof(text.c_str(), &end);
        if (end == nullptr || *end != '\0') {
            throw std::runtime_error("Invalid number literal at position " + std::to_string(pos) + ": " + text);
        }
        if (!std::isfinite(value)) {
            throw std::runtime_error("Number literal is not finite at position " + std::to_string(pos) + ": " + text);
        }

        const std::uint16_t idx = static_cast<std::uint16_t>(parsed.constants.size());
        parsed.constants.push_back(value);
        parsed.rpn.push_back(RpnToken{RpnToken::Kind::Constant, idx, OpCode::Add});
    };

    bool expect_operand = true;

    for (std::size_t i = 0; i < lex.size(); ++i) {
        const auto& tok = lex[i];

        if (tok.kind == LexKind::Identifier) {
            const bool has_lparen_next = (i + 1 < lex.size() && lex[i + 1].kind == LexKind::LParen);
            if (has_lparen_next) {
                if (tok.text != "sin" && tok.text != "sqrt") {
                    throw std::runtime_error(
                        "Unknown function at position " + std::to_string(tok.pos) + ": " + tok.text
                    );
                }

                op_stack.push_back(StackToken{StackToken::Kind::Function, function_from_name(tok.text, tok.pos), tok.pos});
                expect_operand = true;
            } else {
                if (!expect_operand) {
                    throw std::runtime_error("Missing operator before identifier at position " + std::to_string(tok.pos));
                }
                push_variable(tok.text);
                expect_operand = false;
            }
            continue;
        }

        if (tok.kind == LexKind::Number) {
            if (!expect_operand) {
                throw std::runtime_error("Missing operator before number at position " + std::to_string(tok.pos));
            }
            push_constant(tok.text, tok.pos);
            expect_operand = false;
            continue;
        }

        if (tok.kind == LexKind::Operator) {
            if (expect_operand) {
                throw std::runtime_error("Unexpected binary operator at position " + std::to_string(tok.pos));
            }

            OpCode current_op = binary_from_char(tok.text[0], tok.pos);
            while (!op_stack.empty()) {
                const auto top = op_stack.back();
                if (top.kind == StackToken::Kind::LParen) {
                    break;
                }

                if (top.kind == StackToken::Kind::Function) {
                    parsed.rpn.push_back(RpnToken{RpnToken::Kind::UnaryOp, 0, top.op});
                    op_stack.pop_back();
                    continue;
                }

                const int top_prec = precedence(top.op);
                const int cur_prec = precedence(current_op);
                if (top_prec >= cur_prec) {
                    parsed.rpn.push_back(RpnToken{RpnToken::Kind::BinaryOp, 0, top.op});
                    op_stack.pop_back();
                } else {
                    break;
                }
            }

            op_stack.push_back(StackToken{StackToken::Kind::BinaryOp, current_op, tok.pos});
            expect_operand = true;
            continue;
        }

        if (tok.kind == LexKind::LParen) {
            op_stack.push_back(StackToken{StackToken::Kind::LParen, OpCode::Add, tok.pos});
            expect_operand = true;
            continue;
        }

        if (tok.kind == LexKind::RParen) {
            if (expect_operand) {
                throw std::runtime_error("Unexpected ')' at position " + std::to_string(tok.pos));
            }

            bool found_lparen = false;
            while (!op_stack.empty()) {
                const auto top = op_stack.back();
                op_stack.pop_back();

                if (top.kind == StackToken::Kind::LParen) {
                    found_lparen = true;
                    break;
                }

                if (top.kind == StackToken::Kind::Function) {
                    parsed.rpn.push_back(RpnToken{RpnToken::Kind::UnaryOp, 0, top.op});
                } else {
                    parsed.rpn.push_back(RpnToken{RpnToken::Kind::BinaryOp, 0, top.op});
                }
            }

            if (!found_lparen) {
                throw std::runtime_error("Mismatched ')' at position " + std::to_string(tok.pos));
            }

            if (!op_stack.empty() && op_stack.back().kind == StackToken::Kind::Function) {
                const auto fn = op_stack.back();
                op_stack.pop_back();
                parsed.rpn.push_back(RpnToken{RpnToken::Kind::UnaryOp, 0, fn.op});
            }

            expect_operand = false;
            continue;
        }
    }

    if (expect_operand) {
        throw std::runtime_error("Expression ended unexpectedly (expect operand)");
    }

    while (!op_stack.empty()) {
        const auto top = op_stack.back();
        op_stack.pop_back();

        if (top.kind == StackToken::Kind::LParen) {
            throw std::runtime_error("Mismatched '(' at position " + std::to_string(top.pos));
        }

        if (top.kind == StackToken::Kind::Function) {
            parsed.rpn.push_back(RpnToken{RpnToken::Kind::UnaryOp, 0, top.op});
        } else {
            parsed.rpn.push_back(RpnToken{RpnToken::Kind::BinaryOp, 0, top.op});
        }
    }

    if (parsed.rpn.empty()) {
        throw std::runtime_error("Expression parsing generated empty RPN");
    }

    return parsed;
}

} // namespace benchcalc
