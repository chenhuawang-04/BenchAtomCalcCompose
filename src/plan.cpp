#include "benchcalc/plan.h"

#include <stdexcept>
#include <vector>

namespace benchcalc {
namespace {

struct ValueRef {
    std::uint16_t buffer = 0;
    bool can_overwrite = false;
};

} // namespace

ExecutionPlan compile_plan(const ParsedExpression& parsed) {
    ExecutionPlan plan;
    plan.expression = parsed;
    plan.variable_buffers = static_cast<std::uint16_t>(parsed.variables.size());
    plan.constant_buffers = static_cast<std::uint16_t>(parsed.constants.size());

    if (plan.variable_buffers == 0) {
        throw std::runtime_error("Expression has no variables");
    }

    std::vector<int> remaining_use(plan.variable_buffers, 0);
    for (const auto& token : parsed.rpn) {
        if (token.kind == RpnToken::Kind::Variable) {
            ++remaining_use[token.index];
        }
    }

    std::vector<ValueRef> stack;
    stack.reserve(parsed.rpn.size());

    const std::uint16_t constant_base = plan.variable_buffers;
    std::uint16_t next_temp = static_cast<std::uint16_t>(plan.variable_buffers + plan.constant_buffers);

    auto alloc_temp = [&]() -> std::uint16_t {
        return next_temp++;
    };

    auto ensure_writable = [&](ValueRef v) -> ValueRef {
        if (v.can_overwrite) {
            return v;
        }

        const std::uint16_t temp = alloc_temp();
        plan.steps.push_back(PlanStep{StepKind::Copy, OpCode::Add, temp, v.buffer});
        return ValueRef{temp, true};
    };

    for (const auto& token : parsed.rpn) {
        switch (token.kind) {
        case RpnToken::Kind::Variable: {
            if (token.index >= plan.variable_buffers) {
                throw std::runtime_error("Invalid variable index in RPN");
            }

            --remaining_use[token.index];
            const bool last_occurrence = (remaining_use[token.index] == 0);
            stack.push_back(ValueRef{token.index, last_occurrence});
            break;
        }
        case RpnToken::Kind::Constant: {
            if (token.index >= plan.constant_buffers) {
                throw std::runtime_error("Invalid constant index in RPN");
            }
            const std::uint16_t const_buffer = static_cast<std::uint16_t>(constant_base + token.index);
            stack.push_back(ValueRef{const_buffer, false});
            break;
        }
        case RpnToken::Kind::UnaryOp: {
            if (stack.empty()) {
                throw std::runtime_error("Unary op requires one operand");
            }

            ValueRef v = stack.back();
            stack.pop_back();

            v = ensure_writable(v);
            plan.steps.push_back(PlanStep{StepKind::Unary, token.op, v.buffer, 0});
            stack.push_back(ValueRef{v.buffer, true});
            break;
        }
        case RpnToken::Kind::BinaryOp: {
            if (stack.size() < 2) {
                throw std::runtime_error("Binary op requires two operands");
            }

            ValueRef rhs = stack.back();
            stack.pop_back();
            ValueRef lhs = stack.back();
            stack.pop_back();

            lhs = ensure_writable(lhs);
            plan.steps.push_back(PlanStep{StepKind::Binary, token.op, lhs.buffer, rhs.buffer});
            stack.push_back(ValueRef{lhs.buffer, true});
            break;
        }
        }
    }

    if (stack.size() != 1) {
        throw std::runtime_error("Expression RPN compile failed: final stack size is not 1");
    }

    plan.result_buffer = stack.back().buffer;
    plan.total_buffers = next_temp;

    std::uint64_t bytes = 0;
    std::uint64_t ops = 0;

    for (const auto& step : plan.steps) {
        switch (step.kind) {
        case StepKind::Copy:
            bytes += sizeof(float) * 2ull;
            break;
        case StepKind::Unary:
            bytes += sizeof(float) * 2ull;
            ++ops;
            break;
        case StepKind::Binary:
            bytes += sizeof(float) * 3ull;
            ++ops;
            break;
        }
    }

    plan.bytes_per_element = bytes;
    plan.arithmetic_ops_per_element = ops;
    return plan;
}

} // namespace benchcalc
