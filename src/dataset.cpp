#include "benchcalc/dataset.h"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace benchcalc {
namespace {
constexpr float kMinDivisorForDataset = 1e-3f;
}

bool evaluate_rpn_scalar(
    const ParsedExpression& expr,
    const std::vector<float>& variable_values,
    float& out_value
) {
    if (variable_values.size() != expr.variables.size()) {
        return false;
    }

    std::vector<float> stack;
    stack.reserve(expr.rpn.size());

    for (const auto& token : expr.rpn) {
        if (token.kind == RpnToken::Kind::Variable) {
            if (token.index >= variable_values.size()) {
                return false;
            }
            stack.push_back(variable_values[token.index]);
            continue;
        }

        if (token.kind == RpnToken::Kind::Constant) {
            if (token.index >= expr.constants.size()) {
                return false;
            }
            stack.push_back(expr.constants[token.index]);
            continue;
        }

        if (token.kind == RpnToken::Kind::UnaryOp) {
            if (stack.empty()) {
                return false;
            }

            float v = stack.back();
            stack.pop_back();

            float r = 0.0f;
            if (token.op == OpCode::Sin) {
                r = std::sin(v);
            } else if (token.op == OpCode::Sqrt) {
                if (v < 0.0f) {
                    return false;
                }
                r = std::sqrt(v);
            } else {
                return false;
            }

            if (!std::isfinite(r)) {
                return false;
            }

            stack.push_back(r);
            continue;
        }

        if (token.kind == RpnToken::Kind::BinaryOp) {
            if (stack.size() < 2) {
                return false;
            }

            float rhs = stack.back();
            stack.pop_back();
            float lhs = stack.back();
            stack.pop_back();

            float r = 0.0f;
            switch (token.op) {
            case OpCode::Add:
                r = lhs + rhs;
                break;
            case OpCode::Sub:
                r = lhs - rhs;
                break;
            case OpCode::Mul:
                r = lhs * rhs;
                break;
            case OpCode::Div:
                if (std::fabs(rhs) < kMinDivisorForDataset) {
                    return false;
                }
                r = lhs / rhs;
                break;
            default:
                return false;
            }

            if (!std::isfinite(r)) {
                return false;
            }

            stack.push_back(r);
            continue;
        }

        return false;
    }

    if (stack.size() != 1) {
        return false;
    }

    out_value = stack.back();
    return std::isfinite(out_value);
}

DataSet generate_dataset(
    const ParsedExpression& expr,
    std::size_t length,
    std::uint64_t seed,
    std::size_t max_attempts_per_sample
) {
    DataSet data;
    data.length = length;
    data.inputs.resize(expr.variables.size(), std::vector<float>(length));
    data.reference_output.resize(length);

    std::mt19937_64 rng(seed ^ (length * 0x9E3779B97F4A7C15ULL));
    std::uniform_real_distribution<float> dist_mixed(-3.5f, 3.5f);
    std::uniform_real_distribution<float> dist_positive(0.125f, 3.5f);

    std::vector<float> sample(expr.variables.size(), 0.0f);

    for (std::size_t i = 0; i < length; ++i) {
        bool ok = false;
        float out = 0.0f;

        for (std::size_t attempt = 0; attempt < max_attempts_per_sample; ++attempt) {
            for (auto& v : sample) {
                v = dist_mixed(rng);
            }

            if (evaluate_rpn_scalar(expr, sample, out)) {
                ok = true;
                break;
            }
        }

        if (!ok) {
            for (std::size_t attempt = 0; attempt < max_attempts_per_sample * 2; ++attempt) {
                for (auto& v : sample) {
                    v = dist_positive(rng);
                }

                if (evaluate_rpn_scalar(expr, sample, out)) {
                    ok = true;
                    break;
                }
            }
        }

        if (!ok) {
            throw std::runtime_error(
                "Failed to generate finite sample at index " + std::to_string(i) +
                ". Consider simplifying expression domain or increasing attempts."
            );
        }

        for (std::size_t var = 0; var < sample.size(); ++var) {
            data.inputs[var][i] = sample[var];
        }
        data.reference_output[i] = out;
    }

    return data;
}

} // namespace benchcalc
