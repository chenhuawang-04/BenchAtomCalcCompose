#pragma once

#include <cstdint>
#include <string_view>

namespace benchcalc {

enum class OpCode {
    Add,
    Sub,
    Mul,
    Div,
    Sin,
    Sqrt,
};

inline constexpr bool is_unary(OpCode op) noexcept {
    return op == OpCode::Sin || op == OpCode::Sqrt;
}

inline constexpr bool is_binary(OpCode op) noexcept {
    return op == OpCode::Add || op == OpCode::Sub || op == OpCode::Mul || op == OpCode::Div;
}

inline constexpr std::string_view op_to_string(OpCode op) noexcept {
    switch (op) {
    case OpCode::Add: return "+";
    case OpCode::Sub: return "-";
    case OpCode::Mul: return "*";
    case OpCode::Div: return "/";
    case OpCode::Sin: return "sin";
    case OpCode::Sqrt: return "sqrt";
    }
    return "?";
}

} // namespace benchcalc
