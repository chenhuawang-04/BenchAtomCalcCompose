#pragma once

#include "benchcalc/types.h"

#include <string>
#include <vector>

namespace benchcalc {

struct RpnToken {
    enum class Kind {
        Variable,
        Constant,
        UnaryOp,
        BinaryOp,
    };

    Kind kind{};
    std::uint16_t index = 0;  // Variable/Constant 的索引；算子时忽略
    OpCode op = OpCode::Add;
};

struct ParsedExpression {
    std::string original;
    std::vector<std::string> variables;  // 按首次出现顺序
    std::vector<float> constants;        // 字面量常量池
    std::vector<RpnToken> rpn;           // 后缀表达式
};

} // namespace benchcalc
