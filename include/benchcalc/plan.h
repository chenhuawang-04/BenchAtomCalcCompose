#pragma once

#include "benchcalc/expression.h"

#include <cstdint>
#include <vector>

namespace benchcalc {

enum class StepKind {
    Copy,
    Unary,
    Binary,
};

struct PlanStep {
    StepKind kind = StepKind::Copy;
    OpCode op = OpCode::Add;      // Copy 时忽略
    std::uint16_t dst = 0;
    std::uint16_t src = 0;        // Unary 时忽略，Copy/Binary 使用
};

struct ExecutionPlan {
    ParsedExpression expression;
    std::vector<PlanStep> steps;
    std::uint16_t result_buffer = 0;
    std::uint16_t total_buffers = 0;   // = variables + temporaries
    std::uint16_t variable_buffers = 0;
    std::uint16_t constant_buffers = 0;

    // 统计信息（用于带宽/吞吐估算）
    std::uint64_t bytes_per_element = 0;
    std::uint64_t arithmetic_ops_per_element = 0;
};

ExecutionPlan compile_plan(const ParsedExpression& parsed);

} // namespace benchcalc
