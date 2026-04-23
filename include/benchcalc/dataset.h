#pragma once

#include "benchcalc/expression.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace benchcalc {

struct DataSet {
    std::size_t length = 0;
    std::vector<std::vector<float>> inputs;  // [var][index]
    std::vector<float> reference_output;     // 参考结果
};

bool evaluate_rpn_scalar(
    const ParsedExpression& expr,
    const std::vector<float>& variable_values,
    float& out_value
);

DataSet generate_dataset(
    const ParsedExpression& expr,
    std::size_t length,
    std::uint64_t seed,
    std::size_t max_attempts_per_sample = 128
);

} // namespace benchcalc
