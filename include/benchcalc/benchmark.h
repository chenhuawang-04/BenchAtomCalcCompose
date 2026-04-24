#pragma once

#include "benchcalc/dataset.h"
#include "benchcalc/executor.h"
#include "benchcalc/kernels.h"
#include "benchcalc/plan.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace benchcalc {

struct BenchmarkConfig {
    std::string expression;
    std::vector<std::size_t> sizes = { 1u << 10, 1u << 14, 1u << 18 };
    std::vector<std::size_t> block_sizes = { 32, 64, 128, 256 };
    std::size_t warmup_iterations = 6;
    std::size_t measured_iterations = 24;
    double target_case_ms = 0.0;            // >0 时自动估算迭代次数
    std::size_t max_auto_iterations = 256;  // 自动估算上限
    std::size_t dataset_pool_size = 3;      // 轮换数据集，减少固定缓存形态偏置
    std::size_t repeat_runs = 1;            // 外层重复次数（用于稳定性/置信区间分析）
    std::size_t thread_count = 1;           // 执行线程数（>=1）
    bool include_vm_variants = true;        // 是否包含解释执行基线
    bool include_parallel_variants = false; // 是否包含并行调度变体（thread_count>1 时生效）
    bool include_llvm_jit_variants = true;  // 是否包含 LLVM JIT 执行变体
    std::uint64_t seed = 0xC0FFEE123456789ULL;
    Backend backend = Backend::FastMath;
    bool verify = true;
    float abs_tolerance = 1e-4f;
    float rel_tolerance = 1e-4f;
};

struct Statistics {
    double min_ns = 0.0;
    double max_ns = 0.0;
    double mean_ns = 0.0;
    double median_ns = 0.0;
    double p10_ns = 0.0;
    double p90_ns = 0.0;
    double stdev_ns = 0.0;
    double cv_percent = 0.0;
};

struct BenchmarkResult {
    std::string expression;
    std::string backend_name;
    std::string variant_name;
    std::size_t length = 0;
    std::size_t block_size = 0;
    std::size_t repeat_index = 0;

    Statistics stats;
    std::size_t measured_iterations = 0;

    double million_elements_per_sec = 0.0;
    double estimated_gb_per_sec = 0.0;
    double million_ops_per_sec = 0.0;

    bool verified = false;
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;

    double checksum = 0.0;
};

struct BenchmarkRun {
    ExecutionPlan plan;
    std::vector<BenchmarkResult> results;
};

BenchmarkRun run_benchmark_suite(const BenchmarkConfig& cfg);

} // namespace benchcalc
