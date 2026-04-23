#include "benchcalc/llvm_jit.h"

#include <algorithm>
#include <stdexcept>
#include <thread>

namespace benchcalc {

struct LlvmJitProgram::Impl {
    std::size_t variable_count = 0;
};

LlvmJitProgram::LlvmJitProgram() = default;
LlvmJitProgram::LlvmJitProgram(std::unique_ptr<Impl> impl, std::string error)
    : impl_(std::move(impl)), error_(std::move(error)) {}
LlvmJitProgram::~LlvmJitProgram() = default;
LlvmJitProgram::LlvmJitProgram(LlvmJitProgram&&) noexcept = default;
LlvmJitProgram& LlvmJitProgram::operator=(LlvmJitProgram&&) noexcept = default;

bool LlvmJitProgram::valid() const noexcept { return impl_ != nullptr; }
const std::string& LlvmJitProgram::error() const noexcept { return error_; }
std::size_t LlvmJitProgram::variable_count() const noexcept { return impl_ ? impl_->variable_count : 0; }

bool llvm_jit_supported_by_build() noexcept {
    return false;
}

std::string llvm_jit_build_description() {
    return "LLVM JIT disabled in current build (BENCHCALC_ENABLE_LLVM_JIT=OFF or LLVM-C backend unavailable).";
}

LlvmJitProgram build_llvm_jit_program(const ParsedExpression& expr, Backend backend) {
    (void)expr;
    (void)backend;
    return LlvmJitProgram(nullptr, llvm_jit_build_description());
}

void execute_llvm_jit_program(
    const LlvmJitProgram& program,
    LlvmJitMode mode,
    float* out,
    const std::vector<float*>& variable_buffers,
    std::size_t length,
    std::size_t block_size,
    std::size_t thread_count
) {
    (void)mode;
    (void)out;
    (void)variable_buffers;
    (void)length;
    (void)block_size;
    (void)thread_count;

    if (!program.valid()) {
        throw std::runtime_error("LLVM JIT program is not available: " + program.error());
    }

    throw std::runtime_error("LLVM JIT is disabled in this build.");
}

} // namespace benchcalc
