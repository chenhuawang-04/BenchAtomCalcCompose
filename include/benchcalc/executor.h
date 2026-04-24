#pragma once

#include "benchcalc/kernels.h"
#include "benchcalc/plan.h"

#include <string>
#include <vector>

namespace benchcalc {

enum class ScheduleMode {
    StepMajor,
    BlockMajor,
};

enum class DispatchMode {
    SwitchDispatch,
    FunctionPointerDispatch,
    GlobalVoidOffsetDispatch,     // void() + 全局参数流（由调度器设置起始索引）
    GlobalVoidSignatureDispatch,  // void() + 全局管理器（函数用签名查参数区间）
};

enum class ExecutionModel {
    CompiledPlan,   // 预编译 dst/src 索引计划
    RpnElementVM,   // 元素级 RPN 虚拟机（动态解释基线）
    LlvmJitScalar,  // LLVM ORC JIT（表达式级标量 element kernel）
    LlvmJitLoop,    // LLVM ORC JIT（范围 loop kernel）
};

struct ExecutorSpec {
    std::string name;
    ExecutionModel model = ExecutionModel::CompiledPlan;
    ScheduleMode schedule = ScheduleMode::StepMajor;
    DispatchMode dispatch = DispatchMode::SwitchDispatch;
    bool parallel = false;
};

struct RuntimeStep {
    StepKind kind = StepKind::Copy;
    std::uint16_t dst = 0;
    std::uint16_t src = 0;
    std::uint32_t arg_begin = 0;   // 在 global_arg_slots 里的起始位置
    std::uint32_t signature = 0;   // 全局唯一签名（用于 signature dispatch）

    CopyKernelFn copy = nullptr;
    UnaryKernelFn unary = nullptr;
    BinaryKernelFn binary = nullptr;
    void (*global_void_offset)() = nullptr;
    void (*global_void_signature)() = nullptr;
};

struct RuntimeDispatchData {
    std::vector<RuntimeStep> steps;
    std::vector<std::uint16_t> global_arg_slots;       // 线性参数槽位流
    std::vector<std::uint32_t> signature_to_arg_begin; // signature -> arg_begin
};

struct ExecuteRuntimeOptions {
    std::size_t thread_count = 1;
};

std::vector<ExecutorSpec> default_executors(
    bool include_vm_variants,
    bool include_parallel_variants,
    bool include_llvm_jit_variants,
    std::size_t thread_count
);
RuntimeDispatchData bind_runtime_dispatch_data(const ExecutionPlan& plan, const KernelTable& kernels);

void execute_plan(
    const ExecutionPlan& plan,
    const RuntimeDispatchData& runtime,
    const KernelTable& kernels,
    const KernelConfig& kernel_cfg,
    const ExecutorSpec& executor,
    const ExecuteRuntimeOptions& run_opts,
    std::vector<float*>& buffers,
    std::size_t length,
    std::size_t block_size
);

} // namespace benchcalc
