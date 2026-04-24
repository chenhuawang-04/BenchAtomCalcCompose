#include "benchcalc/benchmark.h"
#include "benchcalc/llvm_jit.h"
#include "benchcalc/parser.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace benchcalc {
namespace {

Statistics summarize_ns(const std::vector<double>& samples_ns) {
    if (samples_ns.empty()) {
        return {};
    }

    Statistics s;
    s.min_ns = *std::min_element(samples_ns.begin(), samples_ns.end());
    s.max_ns = *std::max_element(samples_ns.begin(), samples_ns.end());
    s.mean_ns = std::accumulate(samples_ns.begin(), samples_ns.end(), 0.0) / static_cast<double>(samples_ns.size());

    std::vector<double> sorted = samples_ns;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t n = sorted.size();
    if (n % 2 == 0) {
        s.median_ns = 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
    } else {
        s.median_ns = sorted[n / 2];
    }

    const std::size_t p10_idx = static_cast<std::size_t>(0.10 * static_cast<double>(n - 1));
    const std::size_t p90_idx = static_cast<std::size_t>(0.90 * static_cast<double>(n - 1));
    s.p10_ns = sorted[p10_idx];
    s.p90_ns = sorted[p90_idx];

    double var = 0.0;
    for (double v : samples_ns) {
        const double d = v - s.mean_ns;
        var += d * d;
    }
    var /= static_cast<double>(samples_ns.size());
    s.stdev_ns = std::sqrt(var);
    s.cv_percent = (s.mean_ns > 0.0) ? (s.stdev_ns / s.mean_ns * 100.0) : 0.0;

    return s;
}

const char* backend_name(Backend b) {
    switch (b) {
    case Backend::StdLib: return "std";
    case Backend::FastMath: return "fast_math";
    }
    return "unknown";
}

} // namespace

BenchmarkRun run_benchmark_suite(const BenchmarkConfig& cfg) {
    if (cfg.expression.empty()) {
        throw std::runtime_error("BenchmarkConfig.expression is empty");
    }
    if (cfg.sizes.empty()) {
        throw std::runtime_error("BenchmarkConfig.sizes is empty");
    }
    if (cfg.block_sizes.empty()) {
        throw std::runtime_error("BenchmarkConfig.block_sizes is empty");
    }
    if (cfg.measured_iterations == 0) {
        throw std::runtime_error("BenchmarkConfig.measured_iterations must be > 0");
    }
    if (cfg.thread_count == 0) {
        throw std::runtime_error("BenchmarkConfig.thread_count must be > 0");
    }
    if (cfg.repeat_runs == 0) {
        throw std::runtime_error("BenchmarkConfig.repeat_runs must be > 0");
    }

    const ParsedExpression parsed = parse_expression(cfg.expression);
    const ExecutionPlan plan = compile_plan(parsed);

    const KernelConfig kernel_cfg{cfg.backend};
    const KernelTable kernels = make_kernel_table(cfg.backend);
    const bool enable_llvm_jit = cfg.include_llvm_jit_variants && llvm_jit_supported_by_build();

    const auto executors = default_executors(
        cfg.include_vm_variants,
        cfg.include_parallel_variants,
        enable_llvm_jit,
        cfg.thread_count
    );
    const ExecuteRuntimeOptions run_opts{cfg.thread_count};

    LlvmJitProgram llvm_program;
    if (enable_llvm_jit) {
        llvm_program = build_llvm_jit_program(parsed, cfg.backend);
        if (!llvm_program.valid()) {
            throw std::runtime_error("Failed to build LLVM JIT program: " + llvm_program.error());
        }
    }

    std::uint64_t rpn_ops_per_element = 0;
    std::uint64_t rpn_bytes_per_element = 0;
    for (const auto& t : parsed.rpn) {
        if (t.kind == RpnToken::Kind::Variable || t.kind == RpnToken::Kind::Constant) {
            rpn_bytes_per_element += sizeof(float);
        } else {
            ++rpn_ops_per_element;
        }
    }
    rpn_bytes_per_element += sizeof(float); // 最终结果写回

    BenchmarkRun run;
    run.plan = plan;

    for (std::size_t repeat = 0; repeat < cfg.repeat_runs; ++repeat) {
        const std::uint64_t repeat_seed = cfg.seed + (0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(repeat));

        for (std::size_t length : cfg.sizes) {
            std::vector<std::vector<float>> storage(plan.total_buffers, std::vector<float>(length));
            std::vector<float*> pointers(plan.total_buffers, nullptr);
            for (std::size_t i = 0; i < storage.size(); ++i) {
                pointers[i] = storage[i].data();
            }
            for (std::size_t c = 0; c < plan.constant_buffers; ++c) {
                const std::size_t buffer_idx = plan.variable_buffers + c;
                std::fill(storage[buffer_idx].begin(), storage[buffer_idx].end(), parsed.constants[c]);
            }

            std::vector<float*> llvm_var_ptrs(plan.variable_buffers, nullptr);
            for (std::size_t i = 0; i < plan.variable_buffers; ++i) {
                llvm_var_ptrs[i] = storage[i].data();
            }

            std::vector<DataSet> pool;
            pool.reserve(cfg.dataset_pool_size);
            for (std::size_t p = 0; p < std::max<std::size_t>(1, cfg.dataset_pool_size); ++p) {
                pool.push_back(generate_dataset(
                    parsed,
                    length,
                    repeat_seed + static_cast<std::uint64_t>(length) + static_cast<std::uint64_t>(p * 1315423911ULL)
                ));
            }

            for (std::size_t block : cfg.block_sizes) {
                if (block == 0) {
                    throw std::runtime_error("block size cannot be 0");
                }

                for (const auto& exec : executors) {
                    const bool is_llvm_model =
                        (exec.model == ExecutionModel::LlvmJitScalar) ||
                        (exec.model == ExecutionModel::LlvmJitLoop);

                    const RuntimeDispatchData runtime_dispatch =
                        is_llvm_model
                        ? RuntimeDispatchData{}
                        : bind_runtime_dispatch_data(plan, kernels);

                    auto execute_once = [&]() {
                        if (is_llvm_model) {
                            const LlvmJitMode jit_mode =
                                (exec.model == ExecutionModel::LlvmJitLoop)
                                ? LlvmJitMode::LoopKernel
                                : LlvmJitMode::ScalarElement;

                            execute_llvm_jit_program(
                                llvm_program,
                                jit_mode,
                                storage[plan.result_buffer].data(),
                                llvm_var_ptrs,
                                length,
                                block,
                                exec.parallel ? run_opts.thread_count : 1
                            );
                        } else {
                            execute_plan(plan, runtime_dispatch, kernels, kernel_cfg, exec, run_opts, pointers, length, block);
                        }
                    };

                    for (std::size_t w = 0; w < cfg.warmup_iterations; ++w) {
                        const DataSet& current = pool[w % pool.size()];
                        for (std::size_t v = 0; v < plan.variable_buffers; ++v) {
                            std::copy(current.inputs[v].begin(), current.inputs[v].end(), storage[v].begin());
                        }
                        execute_once();
                    }

                    std::size_t measured_iters = cfg.measured_iterations;
                    if (cfg.target_case_ms > 0.0) {
                        std::vector<double> calib;
                        calib.reserve(2);
                        for (std::size_t c = 0; c < 2; ++c) {
                            const DataSet& current = pool[c % pool.size()];
                            for (std::size_t v = 0; v < plan.variable_buffers; ++v) {
                                std::copy(current.inputs[v].begin(), current.inputs[v].end(), storage[v].begin());
                            }

                            const auto t0 = std::chrono::steady_clock::now();
                            execute_once();
                            const auto t1 = std::chrono::steady_clock::now();
                            calib.push_back(static_cast<double>(
                                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
                            ));
                        }
                        const auto cstats = summarize_ns(calib);
                        const double target_ns = cfg.target_case_ms * 1e6;
                        const double one_iter_ns = std::max(cstats.median_ns, 1.0);
                        measured_iters = static_cast<std::size_t>(target_ns / one_iter_ns);
                        measured_iters = std::max<std::size_t>(3, measured_iters);
                        measured_iters = std::min<std::size_t>(cfg.max_auto_iterations, measured_iters);
                    }

                    std::vector<double> ns_samples;
                    ns_samples.reserve(measured_iters);

                    double checksum = 0.0;
                    const DataSet* last_data = nullptr;
                    for (std::size_t iter = 0; iter < measured_iters; ++iter) {
                        const DataSet& current = pool[iter % pool.size()];
                        last_data = &current;

                        for (std::size_t v = 0; v < plan.variable_buffers; ++v) {
                            std::copy(current.inputs[v].begin(), current.inputs[v].end(), storage[v].begin());
                        }

                        const auto t0 = std::chrono::steady_clock::now();
                        execute_once();
                        const auto t1 = std::chrono::steady_clock::now();

                        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                        ns_samples.push_back(static_cast<double>(ns));

                        checksum += static_cast<double>(storage[plan.result_buffer][iter % length]);
                    }

                    BenchmarkResult result;
                    result.expression = cfg.expression;
                    result.backend_name = backend_name(cfg.backend);
                    result.variant_name = exec.name;
                    result.length = length;
                    result.block_size = block;
                    result.repeat_index = repeat;
                    result.measured_iterations = measured_iters;
                    result.stats = summarize_ns(ns_samples);
                    result.checksum = checksum;

                    const double median_sec = result.stats.median_ns * 1e-9;
                    const std::uint64_t ops_per_element = (exec.model == ExecutionModel::CompiledPlan)
                        ? plan.arithmetic_ops_per_element
                        : rpn_ops_per_element;
                    const std::uint64_t bytes_per_element = (exec.model == ExecutionModel::CompiledPlan)
                        ? plan.bytes_per_element
                        : rpn_bytes_per_element;
                    result.million_elements_per_sec = (median_sec > 0.0)
                        ? (static_cast<double>(length) / median_sec) / 1e6
                        : 0.0;

                    result.estimated_gb_per_sec = (median_sec > 0.0)
                        ? (static_cast<double>(bytes_per_element) * static_cast<double>(length) / median_sec) / 1e9
                        : 0.0;

                    result.million_ops_per_sec = (median_sec > 0.0)
                        ? (static_cast<double>(ops_per_element) * static_cast<double>(length) / median_sec) / 1e6
                        : 0.0;

                    if (cfg.verify) {
                        float max_abs = 0.0f;
                        float max_rel = 0.0f;
                        bool ok = true;

                        const auto& out = storage[plan.result_buffer];
                        const auto& ref_data = *last_data;
                        for (std::size_t i = 0; i < length; ++i) {
                            const float ref = ref_data.reference_output[i];
                            const float val = out[i];
                            const float abs_err = std::fabs(ref - val);
                            const float denom = std::max(std::fabs(ref), 1e-8f);
                            const float rel_err = abs_err / denom;

                            max_abs = std::max(max_abs, abs_err);
                            max_rel = std::max(max_rel, rel_err);

                            if (abs_err > cfg.abs_tolerance && rel_err > cfg.rel_tolerance) {
                                ok = false;
                            }
                        }

                        result.verified = ok;
                        result.max_abs_error = max_abs;
                        result.max_rel_error = max_rel;
                    } else {
                        result.verified = true;
                    }

                    run.results.push_back(result);
                }
            }
        }
    }

    return run;
}

} // namespace benchcalc
