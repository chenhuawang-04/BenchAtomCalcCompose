#pragma once

#include "benchcalc/expression.h"
#include "benchcalc/kernels.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace benchcalc {

enum class LlvmJitMode {
    ScalarElement,  // f(vars, idx) -> scalar
    LoopKernel,     // f(vars, out, begin, end) -> range kernel
};

class LlvmJitProgram {
public:
    LlvmJitProgram();
    ~LlvmJitProgram();

    LlvmJitProgram(LlvmJitProgram&&) noexcept;
    LlvmJitProgram& operator=(LlvmJitProgram&&) noexcept;

    LlvmJitProgram(const LlvmJitProgram&) = delete;
    LlvmJitProgram& operator=(const LlvmJitProgram&) = delete;

    bool valid() const noexcept;
    const std::string& error() const noexcept;
    std::size_t variable_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string error_;

    explicit LlvmJitProgram(std::unique_ptr<Impl> impl, std::string error);

    friend LlvmJitProgram build_llvm_jit_program(const ParsedExpression& expr, Backend backend);
    friend void execute_llvm_jit_program(
        const LlvmJitProgram& program,
        LlvmJitMode mode,
        float* out,
        const std::vector<float*>& variable_buffers,
        std::size_t length,
        std::size_t block_size,
        std::size_t thread_count
    );
};

bool llvm_jit_supported_by_build() noexcept;
std::string llvm_jit_build_description();

LlvmJitProgram build_llvm_jit_program(const ParsedExpression& expr, Backend backend);

void execute_llvm_jit_program(
    const LlvmJitProgram& program,
    LlvmJitMode mode,
    float* out,
    const std::vector<float*>& variable_buffers,
    std::size_t length,
    std::size_t block_size,
    std::size_t thread_count
);

} // namespace benchcalc
