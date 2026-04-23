#include "benchcalc/llvm_jit.h"

#if defined(BENCHCALC_HAS_LLVM_JIT)

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include <llvm-c/Analysis.h>
#include <llvm-c/Core.h>
#include <llvm-c/Error.h>
#include <llvm-c/LLJIT.h>
#include <llvm-c/Orc.h>
#include <llvm-c/Target.h>
#include <llvm-c/TargetMachine.h>
#include <llvm/IR/Intrinsics.h>

#if defined(BENCHCALC_HAS_FAST_MATH)
#include <fast_math/sqrt.h>
#include <fast_math/trig.h>
#endif

namespace benchcalc {
namespace {

using JitEvalFn = float(*)(const float* const*, std::uint64_t);
using JitLoopFn = void(*)(const float* const*, float*, std::uint64_t, std::uint64_t);

#if defined(BENCHCALC_HAS_FAST_MATH)
float jit_fast_sin_scalar(float x) {
    return MMath::sin(x);
}

float jit_fast_sqrt_scalar(float x) {
    return MMath::sqrt(x);
}
#endif

std::once_flag g_llvm_init_once;

std::string take_error_message(LLVMErrorRef err) {
    if (!err) {
        return {};
    }
    char* msg = LLVMGetErrorMessage(err);
    std::string out = msg ? msg : "unknown LLVM error";
    LLVMDisposeErrorMessage(msg);
    return out;
}

void ensure_native_target_initialized() {
    std::call_once(g_llvm_init_once, []() {
        if (LLVMInitializeNativeTarget() != 0) {
            throw std::runtime_error("LLVMInitializeNativeTarget failed");
        }
        if (LLVMInitializeNativeAsmPrinter() != 0) {
            throw std::runtime_error("LLVMInitializeNativeAsmPrinter failed");
        }
        if (LLVMInitializeNativeAsmParser() != 0) {
            throw std::runtime_error("LLVMInitializeNativeAsmParser failed");
        }
    });
}

template <typename Fn>
void for_each_chunk(std::size_t chunk_count, std::size_t thread_count, Fn&& fn) {
    if (chunk_count == 0) {
        return;
    }

    if (thread_count <= 1 || chunk_count <= 1) {
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

LLVMValueRef build_intrinsic_call(
    LLVMModuleRef module,
    LLVMBuilderRef builder,
    unsigned intrinsic_id,
    LLVMTypeRef ret_ty,
    LLVMValueRef arg,
    const char* name
) {
    LLVMTypeRef overloaded_types[1] = { ret_ty };
    LLVMValueRef intrinsic_fn = LLVMGetIntrinsicDeclaration(module, intrinsic_id, overloaded_types, 1);
    LLVMTypeRef intrinsic_fn_ty = LLVMGlobalGetValueType(intrinsic_fn);
    LLVMValueRef args[1] = { arg };
    return LLVMBuildCall2(builder, intrinsic_fn_ty, intrinsic_fn, args, 1, name);
}

LLVMValueRef build_variable_value(
    LLVMBuilderRef builder,
    LLVMTypeRef i64_ty,
    LLVMTypeRef ptr_ty,
    LLVMValueRef vars_arg,
    std::uint16_t variable_index,
    const char* slot_ptr_name,
    const char* var_ptr_name
) {
    LLVMValueRef var_slot_idx = LLVMConstInt(i64_ty, variable_index, 0);
    LLVMValueRef var_slot_ptr = LLVMBuildInBoundsGEP2(builder, ptr_ty, vars_arg, &var_slot_idx, 1, slot_ptr_name);
    return LLVMBuildLoad2(builder, ptr_ty, var_slot_ptr, var_ptr_name);
}

LLVMValueRef load_variable_element_from_base(
    LLVMBuilderRef builder,
    LLVMTypeRef float_ty,
    LLVMValueRef var_base_ptr,
    LLVMValueRef idx_arg,
    const char* elem_ptr_name,
    const char* load_name
) {
    LLVMValueRef elem_ptr = LLVMBuildInBoundsGEP2(builder, float_ty, var_base_ptr, &idx_arg, 1, elem_ptr_name);
    return LLVMBuildLoad2(builder, float_ty, elem_ptr, load_name);
}

template <typename LoadVariableFn>
bool emit_expression_value(
    const ParsedExpression& expr,
    LLVMModuleRef module,
    LLVMBuilderRef builder,
    LLVMTypeRef float_ty,
    LLVMTypeRef i64_ty,
    LLVMTypeRef ptr_ty,
    bool use_fast_math_unary,
    LoadVariableFn&& load_variable,
    LLVMValueRef& out_value,
    std::string& out_error
) {
    std::vector<LLVMValueRef> stack;
    stack.reserve(expr.rpn.size());

    for (const auto& token : expr.rpn) {
        if (token.kind == RpnToken::Kind::Variable) {
            if (token.index >= expr.variables.size()) {
                out_error = "LLVM JIT compile failed: variable index out of range.";
                return false;
            }
            stack.push_back(load_variable(token.index));
            continue;
        }

        if (token.kind == RpnToken::Kind::Constant) {
            if (token.index >= expr.constants.size()) {
                out_error = "LLVM JIT compile failed: constant index out of range.";
                return false;
            }
            stack.push_back(LLVMConstReal(float_ty, expr.constants[token.index]));
            continue;
        }

        if (token.kind == RpnToken::Kind::UnaryOp) {
            if (stack.empty()) {
                out_error = "LLVM JIT compile failed: unary stack underflow.";
                return false;
            }

            LLVMValueRef v = stack.back();
            stack.pop_back();

            LLVMValueRef out = nullptr;
            switch (token.op) {
            case OpCode::Sin:
                if (use_fast_math_unary) {
#if defined(BENCHCALC_HAS_FAST_MATH)
                    LLVMTypeRef fast_unary_ty = LLVMFunctionType(float_ty, &float_ty, 1, 0);
                    LLVMValueRef fast_addr = LLVMConstInt(
                        i64_ty,
                        static_cast<unsigned long long>(
                            reinterpret_cast<std::uintptr_t>(&jit_fast_sin_scalar)
                        ),
                        0
                    );
                    LLVMValueRef fast_fn_ptr = LLVMConstIntToPtr(fast_addr, ptr_ty);
                    LLVMValueRef args[1] = { v };
                    out = LLVMBuildCall2(builder, fast_unary_ty, fast_fn_ptr, args, 1, "sin_fast_v");
#else
                    out_error = "LLVM JIT compile failed: fast_math unary requested but BENCHCALC_HAS_FAST_MATH is off.";
                    return false;
#endif
                } else {
                    out = build_intrinsic_call(
                        module,
                        builder,
                        static_cast<unsigned>(llvm::Intrinsic::sin),
                        float_ty,
                        v,
                        "sin_v"
                    );
                }
                break;
            case OpCode::Sqrt:
                if (use_fast_math_unary) {
#if defined(BENCHCALC_HAS_FAST_MATH)
                    LLVMTypeRef fast_unary_ty = LLVMFunctionType(float_ty, &float_ty, 1, 0);
                    LLVMValueRef fast_addr = LLVMConstInt(
                        i64_ty,
                        static_cast<unsigned long long>(
                            reinterpret_cast<std::uintptr_t>(&jit_fast_sqrt_scalar)
                        ),
                        0
                    );
                    LLVMValueRef fast_fn_ptr = LLVMConstIntToPtr(fast_addr, ptr_ty);
                    LLVMValueRef args[1] = { v };
                    out = LLVMBuildCall2(builder, fast_unary_ty, fast_fn_ptr, args, 1, "sqrt_fast_v");
#else
                    out_error = "LLVM JIT compile failed: fast_math unary requested but BENCHCALC_HAS_FAST_MATH is off.";
                    return false;
#endif
                } else {
                    out = build_intrinsic_call(
                        module,
                        builder,
                        static_cast<unsigned>(llvm::Intrinsic::sqrt),
                        float_ty,
                        v,
                        "sqrt_v"
                    );
                }
                break;
            default:
                out_error = "LLVM JIT compile failed: unsupported unary opcode.";
                return false;
            }

            stack.push_back(out);
            continue;
        }

        if (token.kind == RpnToken::Kind::BinaryOp) {
            if (stack.size() < 2) {
                out_error = "LLVM JIT compile failed: binary stack underflow.";
                return false;
            }

            LLVMValueRef rhs = stack.back();
            stack.pop_back();
            LLVMValueRef lhs = stack.back();
            stack.pop_back();

            LLVMValueRef out = nullptr;
            switch (token.op) {
            case OpCode::Add:
                out = LLVMBuildFAdd(builder, lhs, rhs, "add_v");
                break;
            case OpCode::Sub:
                out = LLVMBuildFSub(builder, lhs, rhs, "sub_v");
                break;
            case OpCode::Mul:
                out = LLVMBuildFMul(builder, lhs, rhs, "mul_v");
                break;
            case OpCode::Div:
                out = LLVMBuildFDiv(builder, lhs, rhs, "div_v");
                break;
            default:
                out_error = "LLVM JIT compile failed: unsupported binary opcode.";
                return false;
            }

            stack.push_back(out);
            continue;
        }

        out_error = "LLVM JIT compile failed: unknown token kind.";
        return false;
    }

    if (stack.size() != 1) {
        out_error = "LLVM JIT compile failed: final stack size != 1.";
        return false;
    }

    out_value = stack.back();
    return true;
}

} // namespace

struct LlvmJitProgram::Impl {
    LLVMOrcLLJITRef jit = nullptr;
    JitEvalFn scalar_fn = nullptr;
    JitLoopFn loop_fn = nullptr;
    std::size_t variable_count = 0;

    ~Impl() {
        if (jit != nullptr) {
            LLVMErrorRef err = LLVMOrcDisposeLLJIT(jit);
            (void)err;
            jit = nullptr;
        }
    }
};

LlvmJitProgram::LlvmJitProgram() = default;
LlvmJitProgram::LlvmJitProgram(std::unique_ptr<Impl> impl, std::string error)
    : impl_(std::move(impl)), error_(std::move(error)) {}
LlvmJitProgram::~LlvmJitProgram() = default;
LlvmJitProgram::LlvmJitProgram(LlvmJitProgram&&) noexcept = default;
LlvmJitProgram& LlvmJitProgram::operator=(LlvmJitProgram&&) noexcept = default;

bool LlvmJitProgram::valid() const noexcept { return impl_ != nullptr && impl_->scalar_fn != nullptr && impl_->loop_fn != nullptr; }
const std::string& LlvmJitProgram::error() const noexcept { return error_; }
std::size_t LlvmJitProgram::variable_count() const noexcept { return impl_ ? impl_->variable_count : 0; }

bool llvm_jit_supported_by_build() noexcept {
    return true;
}

std::string llvm_jit_build_description() {
    return "LLVM JIT enabled (LLVM-C ORC LLJIT, scalar+loop kernels, backend-aware unary lowering).";
}

LlvmJitProgram build_llvm_jit_program(const ParsedExpression& expr, Backend backend) {
    try {
        ensure_native_target_initialized();
    }
    catch (const std::exception& ex) {
        return LlvmJitProgram(nullptr, ex.what());
    }

    std::unique_ptr<LlvmJitProgram::Impl> impl = std::make_unique<LlvmJitProgram::Impl>();
    impl->variable_count = expr.variables.size();

    LLVMOrcLLJITBuilderRef jit_builder = LLVMOrcCreateLLJITBuilder();
    LLVMErrorRef err = LLVMOrcCreateLLJIT(&impl->jit, jit_builder);
    if (err != nullptr) {
        return LlvmJitProgram(nullptr, "LLVMOrcCreateLLJIT failed: " + take_error_message(err));
    }

    LLVMContextRef ctx = LLVMContextCreate();
    LLVMModuleRef module = LLVMModuleCreateWithNameInContext("benchcalc_expr_module", ctx);
    LLVMSetDataLayout(module, LLVMOrcLLJITGetDataLayoutStr(impl->jit));
    LLVMSetTarget(module, LLVMOrcLLJITGetTripleString(impl->jit));

    LLVMBuilderRef ir_builder = LLVMCreateBuilderInContext(ctx);
    auto compile_fail = [&](const std::string& message) -> LlvmJitProgram {
        LLVMDisposeBuilder(ir_builder);
        LLVMDisposeModule(module);
        LLVMContextDispose(ctx);
        return LlvmJitProgram(nullptr, message);
    };

    LLVMTypeRef float_ty = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef i64_ty = LLVMInt64TypeInContext(ctx);
    LLVMTypeRef ptr_ty = LLVMPointerTypeInContext(ctx, 0);
    LLVMTypeRef void_ty = LLVMVoidTypeInContext(ctx);
    const bool use_fast_math_unary =
#if defined(BENCHCALC_HAS_FAST_MATH)
        (backend == Backend::FastMath);
#else
        false;
#endif

    // ----------------------------
    // 1) scalar kernel:
    //    float bench_expr_fn(const float* const* vars, uint64_t idx)
    // ----------------------------
    LLVMTypeRef scalar_args[2] = { ptr_ty, i64_ty };
    LLVMTypeRef scalar_fn_ty = LLVMFunctionType(float_ty, scalar_args, 2, 0);
    LLVMValueRef scalar_fn = LLVMAddFunction(module, "bench_expr_fn", scalar_fn_ty);

    LLVMBasicBlockRef scalar_entry = LLVMAppendBasicBlockInContext(ctx, scalar_fn, "scalar_entry");
    LLVMPositionBuilderAtEnd(ir_builder, scalar_entry);

    LLVMValueRef scalar_vars_arg = LLVMGetParam(scalar_fn, 0);
    LLVMValueRef scalar_idx_arg = LLVMGetParam(scalar_fn, 1);
    std::vector<LLVMValueRef> scalar_var_bases(expr.variables.size(), nullptr);
    for (std::size_t vi = 0; vi < expr.variables.size(); ++vi) {
        scalar_var_bases[vi] = build_variable_value(
            ir_builder,
            i64_ty,
            ptr_ty,
            scalar_vars_arg,
            static_cast<std::uint16_t>(vi),
            "scalar_var_slot_ptr",
            "scalar_var_base"
        );
    }

    std::string scalar_error;
    LLVMValueRef scalar_value = nullptr;
    auto scalar_load_variable = [&](std::uint16_t variable_index) -> LLVMValueRef {
        return load_variable_element_from_base(
            ir_builder,
            float_ty,
            scalar_var_bases[variable_index],
            scalar_idx_arg,
            "scalar_elem_ptr",
            "scalar_var_val"
        );
    };

    if (!emit_expression_value(
            expr,
            module,
            ir_builder,
            float_ty,
            i64_ty,
            ptr_ty,
            use_fast_math_unary,
            scalar_load_variable,
            scalar_value,
            scalar_error)) {
        return compile_fail(scalar_error);
    }
    LLVMBuildRet(ir_builder, scalar_value);

    // ----------------------------
    // 2) loop kernel:
    //    void bench_expr_kernel(const float* const* vars, float* out, uint64_t begin, uint64_t end)
    // ----------------------------
    LLVMTypeRef loop_args[4] = { ptr_ty, ptr_ty, i64_ty, i64_ty };
    LLVMTypeRef loop_fn_ty = LLVMFunctionType(void_ty, loop_args, 4, 0);
    LLVMValueRef loop_fn = LLVMAddFunction(module, "bench_expr_kernel", loop_fn_ty);

    LLVMBasicBlockRef loop_entry = LLVMAppendBasicBlockInContext(ctx, loop_fn, "loop_entry");
    LLVMBasicBlockRef loop_check = LLVMAppendBasicBlockInContext(ctx, loop_fn, "loop_check");
    LLVMBasicBlockRef loop_body = LLVMAppendBasicBlockInContext(ctx, loop_fn, "loop_body");
    LLVMBasicBlockRef loop_exit = LLVMAppendBasicBlockInContext(ctx, loop_fn, "loop_exit");

    LLVMValueRef loop_vars_arg = LLVMGetParam(loop_fn, 0);
    LLVMValueRef loop_out_arg = LLVMGetParam(loop_fn, 1);
    LLVMValueRef loop_begin_arg = LLVMGetParam(loop_fn, 2);
    LLVMValueRef loop_end_arg = LLVMGetParam(loop_fn, 3);

    LLVMPositionBuilderAtEnd(ir_builder, loop_entry);
    std::vector<LLVMValueRef> loop_var_bases(expr.variables.size(), nullptr);
    for (std::size_t vi = 0; vi < expr.variables.size(); ++vi) {
        loop_var_bases[vi] = build_variable_value(
            ir_builder,
            i64_ty,
            ptr_ty,
            loop_vars_arg,
            static_cast<std::uint16_t>(vi),
            "loop_var_slot_ptr",
            "loop_var_base"
        );
    }
    LLVMBuildBr(ir_builder, loop_check);

    LLVMPositionBuilderAtEnd(ir_builder, loop_check);

    LLVMValueRef i_phi = LLVMBuildPhi(ir_builder, i64_ty, "i");
    LLVMAddIncoming(i_phi, &loop_begin_arg, &loop_entry, 1);

    LLVMValueRef keep_looping = LLVMBuildICmp(ir_builder, LLVMIntULT, i_phi, loop_end_arg, "loop_cond");
    LLVMBuildCondBr(ir_builder, keep_looping, loop_body, loop_exit);

    LLVMPositionBuilderAtEnd(ir_builder, loop_body);
    std::string loop_error;
    LLVMValueRef loop_value = nullptr;
    auto loop_load_variable = [&](std::uint16_t variable_index) -> LLVMValueRef {
        return load_variable_element_from_base(
            ir_builder,
            float_ty,
            loop_var_bases[variable_index],
            i_phi,
            "loop_elem_ptr",
            "loop_var_val"
        );
    };

    if (!emit_expression_value(
            expr,
            module,
            ir_builder,
            float_ty,
            i64_ty,
            ptr_ty,
            use_fast_math_unary,
            loop_load_variable,
            loop_value,
            loop_error)) {
        return compile_fail(loop_error);
    }

    LLVMValueRef out_elem_ptr = LLVMBuildInBoundsGEP2(ir_builder, float_ty, loop_out_arg, &i_phi, 1, "out_elem_ptr");
    LLVMBuildStore(ir_builder, loop_value, out_elem_ptr);
    LLVMValueRef i_next = LLVMBuildAdd(ir_builder, i_phi, LLVMConstInt(i64_ty, 1, 0), "i_next");
    LLVMBuildBr(ir_builder, loop_check);
    LLVMAddIncoming(i_phi, &i_next, &loop_body, 1);

    LLVMPositionBuilderAtEnd(ir_builder, loop_exit);
    LLVMBuildRetVoid(ir_builder);

    char* verify_message = nullptr;
    if (LLVMVerifyModule(module, LLVMReturnStatusAction, &verify_message) != 0) {
        const std::string msg = verify_message ? verify_message : "unknown verify error";
        LLVMDisposeMessage(verify_message);
        return compile_fail("LLVM IR verify failed: " + msg);
    }

    LLVMDisposeBuilder(ir_builder);

    LLVMOrcThreadSafeContextRef ts_ctx = LLVMOrcCreateNewThreadSafeContextFromLLVMContext(ctx);
    if (ts_ctx == nullptr) {
        LLVMDisposeModule(module);
        LLVMContextDispose(ctx);
        return LlvmJitProgram(nullptr, "LLVMOrcCreateNewThreadSafeContextFromLLVMContext failed");
    }

    LLVMOrcThreadSafeModuleRef tsm = LLVMOrcCreateNewThreadSafeModule(module, ts_ctx);
    if (tsm == nullptr) {
        LLVMDisposeModule(module);
        LLVMOrcDisposeThreadSafeContext(ts_ctx);
        return LlvmJitProgram(nullptr, "LLVMOrcCreateNewThreadSafeModule failed");
    }
    LLVMOrcDisposeThreadSafeContext(ts_ctx);

    LLVMErrorRef add_err = LLVMOrcLLJITAddLLVMIRModule(
        impl->jit,
        LLVMOrcLLJITGetMainJITDylib(impl->jit),
        tsm
    );
    if (add_err != nullptr) {
        return LlvmJitProgram(nullptr, "LLVMOrcLLJITAddLLVMIRModule failed: " + take_error_message(add_err));
    }

    LLVMOrcExecutorAddress scalar_addr = 0;
    LLVMErrorRef scalar_lookup_err = LLVMOrcLLJITLookup(impl->jit, &scalar_addr, "bench_expr_fn");
    if (scalar_lookup_err != nullptr) {
        return LlvmJitProgram(nullptr, "LLVMOrcLLJITLookup(bench_expr_fn) failed: " + take_error_message(scalar_lookup_err));
    }

    LLVMOrcExecutorAddress loop_addr = 0;
    LLVMErrorRef loop_lookup_err = LLVMOrcLLJITLookup(impl->jit, &loop_addr, "bench_expr_kernel");
    if (loop_lookup_err != nullptr) {
        return LlvmJitProgram(nullptr, "LLVMOrcLLJITLookup(bench_expr_kernel) failed: " + take_error_message(loop_lookup_err));
    }

    impl->scalar_fn = reinterpret_cast<JitEvalFn>(static_cast<std::uintptr_t>(scalar_addr));
    impl->loop_fn = reinterpret_cast<JitLoopFn>(static_cast<std::uintptr_t>(loop_addr));

    if (impl->scalar_fn == nullptr) {
        return LlvmJitProgram(nullptr, "LLVM JIT lookup returned null scalar function pointer.");
    }
    if (impl->loop_fn == nullptr) {
        return LlvmJitProgram(nullptr, "LLVM JIT lookup returned null loop function pointer.");
    }

    return LlvmJitProgram(std::move(impl), "");
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
    if (!program.valid()) {
        throw std::runtime_error("LLVM JIT program is invalid: " + program.error());
    }
    if (out == nullptr) {
        throw std::runtime_error("LLVM JIT execute failed: output pointer is null.");
    }

    const std::size_t var_count = program.impl_->variable_count;
    if (variable_buffers.size() < var_count) {
        throw std::runtime_error("LLVM JIT execute failed: variable buffer count is smaller than expression variables.");
    }

    if (length == 0) {
        return;
    }

    std::vector<const float*> vars(var_count, nullptr);
    for (std::size_t i = 0; i < var_count; ++i) {
        vars[i] = variable_buffers[i];
        if (vars[i] == nullptr) {
            throw std::runtime_error("LLVM JIT execute failed: null variable buffer pointer.");
        }
    }

    const std::size_t chunk = (block_size == 0) ? length : block_size;
    const std::size_t chunk_count = (length + chunk - 1) / chunk;
    const std::size_t threads = std::max<std::size_t>(1, thread_count);

    const JitEvalFn scalar_fn = program.impl_->scalar_fn;
    const JitLoopFn loop_fn = program.impl_->loop_fn;
    if (mode == LlvmJitMode::ScalarElement && scalar_fn == nullptr) {
        throw std::runtime_error("LLVM JIT execute failed: scalar function pointer is null.");
    }
    if (mode == LlvmJitMode::LoopKernel && loop_fn == nullptr) {
        throw std::runtime_error("LLVM JIT execute failed: loop function pointer is null.");
    }

    for_each_chunk(chunk_count, threads, [&](std::size_t chunk_index) {
        const std::size_t offset = chunk_index * chunk;
        const std::size_t n = std::min(chunk, length - offset);
        const std::size_t end = offset + n;

        if (mode == LlvmJitMode::LoopKernel) {
            loop_fn(
                vars.data(),
                out,
                static_cast<std::uint64_t>(offset),
                static_cast<std::uint64_t>(end)
            );
            return;
        }

        for (std::size_t idx = offset; idx < end; ++idx) {
            out[idx] = scalar_fn(vars.data(), static_cast<std::uint64_t>(idx));
        }
    });
}

} // namespace benchcalc

#endif
