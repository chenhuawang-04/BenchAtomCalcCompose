#pragma once

#include "benchcalc/expression.h"

#include <string_view>

namespace benchcalc {

ParsedExpression parse_expression(std::string_view expression);

} // namespace benchcalc
