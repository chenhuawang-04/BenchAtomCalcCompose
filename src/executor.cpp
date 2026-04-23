#include "benchcalc/executor.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(BENCHCALC_HAS_FAST_MATH)
#include <fast_math/sqrt.h>
#include <fast_math/trig.h>
#endif

namespace benchcalc {
namespace {

// ============================
// 标准内核（逐元素）
// ============================

void k_copy(float* dst, const float* src, std::size_t n, const KernelConfig&) {
    std::memcpy(dst, src, n * sizeof(float));
}

void k_add(float* dst, const float* rhs, std::size_t n, const KernelConfig&) {
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] += rhs[i];
    }
}

void k_sub(float* dst, const float* rhs, std::size_t n, const KernelConfig&) {
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] -= rhs[i];
    }
}

void k_mul(float* dst, const float* rhs, std::size_t n, const KernelConfig&) {
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] *= rhs[i];
    }
}

void k_div(float* dst, const float* rhs, std::size_t n, const KernelConfig&) {
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] /= rhs[i];
    }
}

void k_sin_std(float* dst, std::size_t n, const KernelConfig&) {
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = std::sin(dst[i]);
    }
}

void k_sqrt_std(float* dst, std::size_t n, const KernelConfig&) {
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = std::sqrt(dst[i]);
    }
}

#if defined(BENCHCALC_HAS_FAST_MATH)
void k_sin_fast(float* dst, std::size_t n, const KernelConfig&) {
    MMath::sinArray(dst, dst, static_cast<int32_t>(n));
}

void k_sqrt_fast(float* dst, std::size_t n, const KernelConfig&) {
    MMath::sqrtArray(dst, static_cast<int32_t>(n));
}
#endif

float scalar_add(float a, float b) { return a + b; }
float scalar_sub(float a, float b) { return a - b; }
float scalar_mul(float a, float b) { return a * b; }
float scalar_div(float a, float b) { return a / b; }

float scalar_sin_dispatch(float x, const KernelConfig& cfg) {
#if defined(BENCHCALC_HAS_FAST_MATH)
    if (cfg.backend == Backend::FastMath) {
        return MMath::sin(x);
    }
#endif
    return std::sin(x);
}

float scalar_sqrt_dispatch(float x, const KernelConfig& cfg) {
#if defined(BENCHCALC_HAS_FAST_MATH)
    if (cfg.backend == Backend::FastMath) {
        return MMath::sqrt(x);
    }
#endif
    return std::sqrt(x);
}

UnaryKernelFn unary_from_op(OpCode op, const KernelTable& kernels) {
    switch (op) {
    case OpCode::Sin: return kernels.sin;
    case OpCode::Sqrt: return kernels.sqrt;
    default: break;
    }
    return nullptr;
}

BinaryKernelFn binary_from_op(OpCode op, const KernelTable& kernels) {
    switch (op) {
    case OpCode::Add: return kernels.add;
    case OpCode::Sub: return kernels.sub;
    case OpCode::Mul: return kernels.mul;
    case OpCode::Div: return kernels.div;
    default: break;
    }
    return nullptr;
}

void run_step_with_switch(
    const PlanStep& step,
    const KernelTable& kernels,
    const KernelConfig& cfg,
    float* dst,
    const float* src,
    std::size_t count
) {
    switch (step.kind) {
    case StepKind::Copy:
        kernels.copy(dst, src, count, cfg);
        return;
    case StepKind::Unary:
        switch (step.op) {
        case OpCode::Sin:
            kernels.sin(dst, count, cfg);
            return;
        case OpCode::Sqrt:
            kernels.sqrt(dst, count, cfg);
            return;
        default:
            throw std::runtime_error("Invalid unary op in switch dispatcher");
        }
    case StepKind::Binary:
        switch (step.op) {
        case OpCode::Add:
            kernels.add(dst, src, count, cfg);
            return;
        case OpCode::Sub:
            kernels.sub(dst, src, count, cfg);
            return;
        case OpCode::Mul:
            kernels.mul(dst, src, count, cfg);
            return;
        case OpCode::Div:
            kernels.div(dst, src, count, cfg);
            return;
        default:
            throw std::runtime_error("Invalid binary op in switch dispatcher");
        }
    }

    throw std::runtime_error("Unknown step kind in switch dispatcher");
}

void run_step_with_fn(
    const RuntimeStep& step,
    const KernelConfig& cfg,
    float* dst,
    const float* src,
    std::size_t count
) {
    switch (step.kind) {
    case StepKind::Copy:
        step.copy(dst, src, count, cfg);
        return;
    case StepKind::Unary:
        step.unary(dst, count, cfg);
        return;
    case StepKind::Binary:
        step.binary(dst, src, count, cfg);
        return;
    }

    throw std::runtime_error("Unknown step kind in function-pointer dispatcher");
}

template <typename Fn>
void for_each_chunk(std::size_t chunk_count, bool parallel, std::size_t thread_count, Fn&& fn) {
    if (chunk_count == 0) {
        return;
    }

    if (!parallel || thread_count <= 1 || chunk_count <= 1) {
        for (std::size_t i = 0; i < chunk_count; ++i) {
            fn(i);
        }
        return;
    }

    const std::size_t workers = std::min(thread_count, chunk_count);
    std::atomic<std::size_t> next{0};

    auto worker_proc = [&]() {
        while (true) {
            const std::size_t idx = next.fetch_add(1, std::memory_order_relaxed);
            if (idx >= chunk_count) {
                return;
            }
            fn(idx);
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(workers - 1);
    for (std::size_t t = 1; t < workers; ++t) {
        pool.emplace_back(worker_proc);
    }

    worker_proc();

    for (auto& th : pool) {
        th.join();
    }
}

} // namespace

KernelTable make_kernel_table(Backend backend) {
    KernelTable table;
    table.copy = &k_copy;
    table.add = &k_add;
    table.sub = &k_sub;
    table.mul = &k_mul;
    table.div = &k_div;

#if defined(BENCHCALC_HAS_FAST_MATH)
    if (backend == Backend::FastMath) {
        table.sin = &k_sin_fast;
        table.sqrt = &k_sqrt_fast;
    } else {
        table.sin = &k_sin_std;
        table.sqrt = &k_sqrt_std;
    }
#else
    (void)backend;
    table.sin = &k_sin_std;
    table.sqrt = &k_sqrt_std;
#endif

    return table;
}

std::vector<ExecutorSpec> default_executors(
    bool include_vm_variants,
    bool include_parallel_variants,
    bool include_llvm_jit_variants,
    std::size_t thread_count
) {
    std::vector<ExecutorSpec> out = {
        {"step-major/switch", ExecutionModel::CompiledPlan, ScheduleMode::StepMajor, DispatchMode::SwitchDispatch, false},
        {"step-major/fnptr", ExecutionModel::CompiledPlan, ScheduleMode::StepMajor, DispatchMode::FunctionPointerDispatch, false},
        {"block-major/switch", ExecutionModel::CompiledPlan, ScheduleMode::BlockMajor, DispatchMode::SwitchDispatch, false},
        {"block-major/fnptr", ExecutionModel::CompiledPlan, ScheduleMode::BlockMajor, DispatchMode::FunctionPointerDispatch, false},
    };

    if (include_vm_variants) {
        out.push_back({"rpn-vm/switch", ExecutionModel::RpnElementVM, ScheduleMode::BlockMajor, DispatchMode::SwitchDispatch, false});
        out.push_back({"rpn-vm/fnptr", ExecutionModel::RpnElementVM, ScheduleMode::BlockMajor, DispatchMode::FunctionPointerDispatch, false});
    }

    if (include_llvm_jit_variants) {
        out.push_back({"llvm-jit/scalar", ExecutionModel::LlvmJitScalar, ScheduleMode::BlockMajor, DispatchMode::FunctionPointerDispatch, false});
        out.push_back({"llvm-jit/loop", ExecutionModel::LlvmJitLoop, ScheduleMode::BlockMajor, DispatchMode::FunctionPointerDispatch, false});
    }

    if (include_parallel_variants && thread_count > 1) {
        out.push_back({"step-major/switch/mt", ExecutionModel::CompiledPlan, ScheduleMode::StepMajor, DispatchMode::SwitchDispatch, true});
        out.push_back({"step-major/fnptr/mt", ExecutionModel::CompiledPlan, ScheduleMode::StepMajor, DispatchMode::FunctionPointerDispatch, true});
        out.push_back({"block-major/switch/mt", ExecutionModel::CompiledPlan, ScheduleMode::BlockMajor, DispatchMode::SwitchDispatch, true});
        out.push_back({"block-major/fnptr/mt", ExecutionModel::CompiledPlan, ScheduleMode::BlockMajor, DispatchMode::FunctionPointerDispatch, true});
        if (include_llvm_jit_variants) {
            out.push_back({"llvm-jit/scalar/mt", ExecutionModel::LlvmJitScalar, ScheduleMode::BlockMajor, DispatchMode::FunctionPointerDispatch, true});
            out.push_back({"llvm-jit/loop/mt", ExecutionModel::LlvmJitLoop, ScheduleMode::BlockMajor, DispatchMode::FunctionPointerDispatch, true});
        }
    }

    return out;
}

std::vector<RuntimeStep> bind_runtime_steps(const ExecutionPlan& plan, const KernelTable& kernels) {
    std::vector<RuntimeStep> runtime;
    runtime.reserve(plan.steps.size());

    for (const auto& s : plan.steps) {
        RuntimeStep r;
        r.kind = s.kind;
        r.dst = s.dst;
        r.src = s.src;

        switch (s.kind) {
        case StepKind::Copy:
            r.copy = kernels.copy;
            break;
        case StepKind::Unary:
            r.unary = unary_from_op(s.op, kernels);
            if (r.unary == nullptr) {
                throw std::runtime_error("Failed to bind unary kernel");
            }
            break;
        case StepKind::Binary:
            r.binary = binary_from_op(s.op, kernels);
            if (r.binary == nullptr) {
                throw std::runtime_error("Failed to bind binary kernel");
            }
            break;
        }

        runtime.push_back(r);
    }

    return runtime;
}

void execute_plan(
    const ExecutionPlan& plan,
    const std::vector<RuntimeStep>& runtime_steps,
    const KernelTable& kernels,
    const KernelConfig& kernel_cfg,
    const ExecutorSpec& executor,
    const ExecuteRuntimeOptions& run_opts,
    std::vector<float*>& buffers,
    std::size_t length,
    std::size_t block_size
) {
    if (buffers.size() < plan.total_buffers) {
        throw std::runtime_error("Buffer pointer list smaller than plan.total_buffers");
    }

    if (length == 0) {
        return;
    }

    const std::size_t chunk = (block_size == 0) ? length : block_size;
    const std::size_t chunk_count = (length + chunk - 1) / chunk;
    const bool use_parallel = executor.parallel && run_opts.thread_count > 1;

    if (executor.model == ExecutionModel::RpnElementVM) {
        using ScalarUnaryFn = float(*)(float, const KernelConfig&);
        using ScalarBinaryFn = float(*)(float, float);

        ScalarUnaryFn sin_fn = &scalar_sin_dispatch;
        ScalarUnaryFn sqrt_fn = &scalar_sqrt_dispatch;
        ScalarBinaryFn add_fn = &scalar_add;
        ScalarBinaryFn sub_fn = &scalar_sub;
        ScalarBinaryFn mul_fn = &scalar_mul;
        ScalarBinaryFn div_fn = &scalar_div;

        for_each_chunk(chunk_count, use_parallel, run_opts.thread_count, [&](std::size_t chunk_index) {
            const std::size_t offset = chunk_index * chunk;
            const std::size_t n = std::min(chunk, length - offset);

            thread_local std::vector<float> stack;
            stack.clear();
            stack.reserve(plan.expression.rpn.size());

            for (std::size_t i = 0; i < n; ++i) {
                const std::size_t index = offset + i;
                stack.clear();

                for (const auto& token : plan.expression.rpn) {
                    if (token.kind == RpnToken::Kind::Variable) {
                        stack.push_back(buffers[token.index][index]);
                        continue;
                    }

                    if (token.kind == RpnToken::Kind::Constant) {
                        const std::uint16_t const_buffer = static_cast<std::uint16_t>(plan.variable_buffers + token.index);
                        stack.push_back(buffers[const_buffer][index]);
                        continue;
                    }

                    if (token.kind == RpnToken::Kind::UnaryOp) {
                        if (stack.empty()) {
                            throw std::runtime_error("RPN VM stack underflow (unary)");
                        }
                        const float v = stack.back();
                        stack.pop_back();

                        float r = 0.0f;
                        if (executor.dispatch == DispatchMode::SwitchDispatch) {
                            switch (token.op) {
                            case OpCode::Sin: r = scalar_sin_dispatch(v, kernel_cfg); break;
                            case OpCode::Sqrt: r = scalar_sqrt_dispatch(v, kernel_cfg); break;
                            default: throw std::runtime_error("RPN VM invalid unary opcode");
                            }
                        } else {
                            ScalarUnaryFn fn = nullptr;
                            switch (token.op) {
                            case OpCode::Sin: fn = sin_fn; break;
                            case OpCode::Sqrt: fn = sqrt_fn; break;
                            default: break;
                            }
                            if (fn == nullptr) {
                                throw std::runtime_error("RPN VM failed to bind unary function");
                            }
                            r = fn(v, kernel_cfg);
                        }
                        stack.push_back(r);
                        continue;
                    }

                    if (token.kind == RpnToken::Kind::BinaryOp) {
                        if (stack.size() < 2) {
                            throw std::runtime_error("RPN VM stack underflow (binary)");
                        }
                        const float rhs = stack.back();
                        stack.pop_back();
                        const float lhs = stack.back();
                        stack.pop_back();

                        float r = 0.0f;
                        if (executor.dispatch == DispatchMode::SwitchDispatch) {
                            switch (token.op) {
                            case OpCode::Add: r = lhs + rhs; break;
                            case OpCode::Sub: r = lhs - rhs; break;
                            case OpCode::Mul: r = lhs * rhs; break;
                            case OpCode::Div: r = lhs / rhs; break;
                            default: throw std::runtime_error("RPN VM invalid binary opcode");
                            }
                        } else {
                            ScalarBinaryFn fn = nullptr;
                            switch (token.op) {
                            case OpCode::Add: fn = add_fn; break;
                            case OpCode::Sub: fn = sub_fn; break;
                            case OpCode::Mul: fn = mul_fn; break;
                            case OpCode::Div: fn = div_fn; break;
                            default: break;
                            }
                            if (fn == nullptr) {
                                throw std::runtime_error("RPN VM failed to bind binary function");
                            }
                            r = fn(lhs, rhs);
                        }
                        stack.push_back(r);
                        continue;
                    }
                }

                if (stack.size() != 1) {
                    throw std::runtime_error("RPN VM evaluation failed: final stack size != 1");
                }
                buffers[plan.result_buffer][index] = stack.back();
            }
        });
        return;
    }

    if (executor.model == ExecutionModel::LlvmJitScalar || executor.model == ExecutionModel::LlvmJitLoop) {
        throw std::runtime_error("execute_plan called with LLVM JIT model. Use execute_llvm_jit_program instead.");
    }

    if (plan.steps.empty()) {
        return;
    }

    if (executor.schedule == ScheduleMode::StepMajor) {
        if (executor.dispatch == DispatchMode::SwitchDispatch) {
            for (const auto& step : plan.steps) {
                float* dst = buffers[step.dst];
                const float* src = (step.kind == StepKind::Unary) ? nullptr : buffers[step.src];

                for_each_chunk(chunk_count, use_parallel, run_opts.thread_count, [&](std::size_t chunk_index) {
                    const std::size_t offset = chunk_index * chunk;
                    const std::size_t n = std::min(chunk, length - offset);
                    run_step_with_switch(step, kernels, kernel_cfg, dst + offset, src ? (src + offset) : nullptr, n);
                });
            }
            return;
        }

        for (const auto& step : runtime_steps) {
            float* dst = buffers[step.dst];
            const float* src = (step.kind == StepKind::Unary) ? nullptr : buffers[step.src];

            for_each_chunk(chunk_count, use_parallel, run_opts.thread_count, [&](std::size_t chunk_index) {
                const std::size_t offset = chunk_index * chunk;
                const std::size_t n = std::min(chunk, length - offset);
                run_step_with_fn(step, kernel_cfg, dst + offset, src ? (src + offset) : nullptr, n);
            });
        }
        return;
    }

    if (executor.dispatch == DispatchMode::SwitchDispatch) {
        for_each_chunk(chunk_count, use_parallel, run_opts.thread_count, [&](std::size_t chunk_index) {
            const std::size_t offset = chunk_index * chunk;
            const std::size_t n = std::min(chunk, length - offset);

            for (const auto& step : plan.steps) {
                float* dst = buffers[step.dst] + offset;
                const float* src = (step.kind == StepKind::Unary) ? nullptr : (buffers[step.src] + offset);
                run_step_with_switch(step, kernels, kernel_cfg, dst, src, n);
            }
        });
        return;
    }

    for_each_chunk(chunk_count, use_parallel, run_opts.thread_count, [&](std::size_t chunk_index) {
        const std::size_t offset = chunk_index * chunk;
        const std::size_t n = std::min(chunk, length - offset);

        for (const auto& step : runtime_steps) {
            float* dst = buffers[step.dst] + offset;
            const float* src = (step.kind == StepKind::Unary) ? nullptr : (buffers[step.src] + offset);
            run_step_with_fn(step, kernel_cfg, dst, src, n);
        }
    });
}

} // namespace benchcalc
