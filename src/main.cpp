#include "benchcalc/benchmark.h"
#include "benchcalc/llvm_jit.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

std::vector<std::size_t> parse_size_list(const std::string& s) {
    std::vector<std::size_t> out;
    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        std::size_t value = static_cast<std::size_t>(std::stoull(item));
        if (value == 0) {
            throw std::runtime_error("List contains 0, which is invalid");
        }
        out.push_back(value);
    }

    if (out.empty()) {
        throw std::runtime_error("Empty list: " + s);
    }

    return out;
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            out += c;
            break;
        }
    }
    return out;
}

void print_usage() {
    std::cout
        << "benchcalc - dynamic expression benchmark\n\n"
        << "Usage:\n"
        << "  benchcalc --expr \"a+b*c/d\" [options]\n\n"
        << "Options:\n"
        << "  --sizes 1024,16384,262144     Vector lengths\n"
        << "  --blocks 32,64,128,256        Block sizes\n"
        << "  --warmup 6                    Warmup iterations\n"
        << "  --iters 24                    Measured iterations\n"
        << "  --target-ms 120               Auto-fit iterations per case to target milliseconds\n"
        << "  --max-auto-iters 256          Upper bound when --target-ms is enabled\n"
        << "  --datasets 3                  Dataset pool size (rotating inputs)\n"
        << "  --threads 1                   Worker threads for parallel variants\n"
        << "  --parallel-variants           Include /mt variants (requires --threads > 1)\n"
        << "  --no-vm                       Exclude RPN VM baseline variants\n"
        << "  --no-llvm-jit                 Exclude LLVM JIT variants\n"
        << "  --seed 12345                  RNG seed\n"
        << "  --backend fast|std            Unary math backend\n"
        << "  --abs-tol 1e-4                Absolute error tolerance for verify\n"
        << "  --rel-tol 1e-4                Relative error tolerance for verify\n"
        << "  --no-verify                   Disable correctness check\n"
        << "  --csv result.csv              Save CSV\n"
        << "  --json result.json            Save JSON\n"
        << "  --dump-plan                   Print compiled execution plan\n"
        << "  --help                        Show this help\n";
}

void save_csv(const std::string& path, const benchcalc::BenchmarkRun& run) {
    std::ofstream ofs(path);
    if (!ofs) {
        throw std::runtime_error("Failed to open CSV file: " + path);
    }

    ofs << "expression,backend,variant,length,block,iters,median_ns,p10_ns,p90_ns,mean_ns,min_ns,max_ns,stdev_ns,cv_percent,melem_per_sec,mops_per_sec,est_gb_per_sec,verified,max_abs_err,max_rel_err,checksum\n";
    for (const auto& r : run.results) {
        ofs
            << '"' << r.expression << "\","
            << r.backend_name << ","
            << r.variant_name << ","
            << r.length << ","
            << r.block_size << ","
            << r.measured_iterations << ","
            << r.stats.median_ns << ","
            << r.stats.p10_ns << ","
            << r.stats.p90_ns << ","
            << r.stats.mean_ns << ","
            << r.stats.min_ns << ","
            << r.stats.max_ns << ","
            << r.stats.stdev_ns << ","
            << r.stats.cv_percent << ","
            << r.million_elements_per_sec << ","
            << r.million_ops_per_sec << ","
            << r.estimated_gb_per_sec << ","
            << (r.verified ? 1 : 0) << ","
            << r.max_abs_error << ","
            << r.max_rel_error << ","
            << r.checksum
            << "\n";
    }
}

void save_json(const std::string& path, const benchcalc::BenchmarkConfig& cfg, const benchcalc::BenchmarkRun& run) {
    std::ofstream ofs(path);
    if (!ofs) {
        throw std::runtime_error("Failed to open JSON file: " + path);
    }

    ofs << "{\n";
    ofs << "  \"expression\": \"" << json_escape(cfg.expression) << "\",\n";
    ofs << "  \"backend\": \"" << (cfg.backend == benchcalc::Backend::FastMath ? "fast_math" : "std") << "\",\n";
    ofs << "  \"thread_count\": " << cfg.thread_count << ",\n";
    ofs << "  \"llvm_jit_build_enabled\": " << (benchcalc::llvm_jit_supported_by_build() ? "true" : "false") << ",\n";
    ofs << "  \"llvm_jit_variants_enabled\": " << (cfg.include_llvm_jit_variants ? "true" : "false") << ",\n";
    ofs << "  \"plan\": {\n";
    ofs << "    \"variables\": " << run.plan.variable_buffers << ",\n";
    ofs << "    \"constants\": " << run.plan.constant_buffers << ",\n";
    ofs << "    \"buffers\": " << run.plan.total_buffers << ",\n";
    ofs << "    \"steps\": " << run.plan.steps.size() << "\n";
    ofs << "  },\n";
    ofs << "  \"results\": [\n";

    for (std::size_t i = 0; i < run.results.size(); ++i) {
        const auto& r = run.results[i];
        ofs << "    {\n";
        ofs << "      \"variant\": \"" << json_escape(r.variant_name) << "\",\n";
        ofs << "      \"length\": " << r.length << ",\n";
        ofs << "      \"block\": " << r.block_size << ",\n";
        ofs << "      \"iters\": " << r.measured_iterations << ",\n";
        ofs << "      \"median_ns\": " << r.stats.median_ns << ",\n";
        ofs << "      \"p10_ns\": " << r.stats.p10_ns << ",\n";
        ofs << "      \"p90_ns\": " << r.stats.p90_ns << ",\n";
        ofs << "      \"cv_percent\": " << r.stats.cv_percent << ",\n";
        ofs << "      \"melem_per_sec\": " << r.million_elements_per_sec << ",\n";
        ofs << "      \"mops_per_sec\": " << r.million_ops_per_sec << ",\n";
        ofs << "      \"est_gb_per_sec\": " << r.estimated_gb_per_sec << ",\n";
        ofs << "      \"verified\": " << (r.verified ? "true" : "false") << ",\n";
        ofs << "      \"max_abs_error\": " << r.max_abs_error << ",\n";
        ofs << "      \"max_rel_error\": " << r.max_rel_error << "\n";
        ofs << "    }" << (i + 1 < run.results.size() ? "," : "") << "\n";
    }

    ofs << "  ]\n";
    ofs << "}\n";
}

void print_best_case_summary(const benchcalc::BenchmarkRun& run) {
    using Key = std::pair<std::size_t, std::size_t>; // length, block
    std::map<Key, std::vector<const benchcalc::BenchmarkResult*>> buckets;

    for (const auto& r : run.results) {
        buckets[{r.length, r.block_size}].push_back(&r);
    }

    std::cout << "\nBest variant by case:\n";
    std::cout << std::left
              << std::setw(10) << "N"
              << std::setw(8) << "Block"
              << std::setw(28) << "Best Variant"
              << std::setw(14) << "Median(ns)"
              << std::setw(10) << "Verify"
              << "\n";
    std::cout << std::string(70, '-') << "\n";

    for (const auto& kv : buckets) {
        const auto& vec = kv.second;

        const benchcalc::BenchmarkResult* best_verified = nullptr;
        const benchcalc::BenchmarkResult* best_any = nullptr;

        for (const auto* r : vec) {
            if (best_any == nullptr || r->stats.median_ns < best_any->stats.median_ns) {
                best_any = r;
            }
            if (r->verified && (best_verified == nullptr || r->stats.median_ns < best_verified->stats.median_ns)) {
                best_verified = r;
            }
        }

        const auto* best = (best_verified != nullptr) ? best_verified : best_any;
        std::cout << std::left
                  << std::setw(10) << kv.first.first
                  << std::setw(8) << kv.first.second
                  << std::setw(28) << best->variant_name
                  << std::setw(14) << std::fixed << std::setprecision(1) << best->stats.median_ns
                  << std::setw(10) << (best->verified ? "OK" : "FAIL")
                  << "\n";
    }
}

void print_baseline_speedup(const benchcalc::BenchmarkRun& run, const std::string& baseline_name) {
    using Key = std::pair<std::size_t, std::size_t>; // length, block

    std::map<Key, const benchcalc::BenchmarkResult*> baseline;
    std::map<std::string, std::vector<double>> speedups;

    for (const auto& r : run.results) {
        if (r.variant_name == baseline_name) {
            baseline[{r.length, r.block_size}] = &r;
        }
    }

    for (const auto& r : run.results) {
        const auto it = baseline.find({r.length, r.block_size});
        if (it == baseline.end()) {
            continue;
        }
        const auto* b = it->second;
        if (b->stats.median_ns <= 0.0 || r.stats.median_ns <= 0.0) {
            continue;
        }
        speedups[r.variant_name].push_back(b->stats.median_ns / r.stats.median_ns);
    }

    std::cout << "\nSpeedup vs baseline [" << baseline_name << "] (arithmetic mean):\n";
    std::cout << std::left << std::setw(28) << "Variant" << std::setw(12) << "Speedup" << "\n";
    std::cout << std::string(40, '-') << "\n";

    for (const auto& kv : speedups) {
        const auto& vals = kv.second;
        double mean = 0.0;
        for (double v : vals) {
            mean += v;
        }
        mean /= static_cast<double>(vals.size());
        std::cout << std::left << std::setw(28) << kv.first << std::setw(12) << std::fixed << std::setprecision(3) << mean << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        benchcalc::BenchmarkConfig cfg;
        std::string csv_path;
        std::string json_path;
        bool dump_plan = false;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];

            auto need_value = [&](const std::string& name) -> std::string {
                if (i + 1 >= argc) {
                    throw std::runtime_error("Missing value for " + name);
                }
                return argv[++i];
            };

            if (arg == "--help" || arg == "-h") {
                print_usage();
                return 0;
            }
            if (arg == "--expr") {
                cfg.expression = need_value("--expr");
                continue;
            }
            if (arg == "--sizes") {
                cfg.sizes = parse_size_list(need_value("--sizes"));
                continue;
            }
            if (arg == "--blocks") {
                cfg.block_sizes = parse_size_list(need_value("--blocks"));
                continue;
            }
            if (arg == "--warmup") {
                cfg.warmup_iterations = static_cast<std::size_t>(std::stoull(need_value("--warmup")));
                continue;
            }
            if (arg == "--iters") {
                cfg.measured_iterations = static_cast<std::size_t>(std::stoull(need_value("--iters")));
                continue;
            }
            if (arg == "--target-ms") {
                cfg.target_case_ms = std::stod(need_value("--target-ms"));
                continue;
            }
            if (arg == "--max-auto-iters") {
                cfg.max_auto_iterations = static_cast<std::size_t>(std::stoull(need_value("--max-auto-iters")));
                continue;
            }
            if (arg == "--datasets") {
                cfg.dataset_pool_size = static_cast<std::size_t>(std::stoull(need_value("--datasets")));
                continue;
            }
            if (arg == "--threads") {
                cfg.thread_count = static_cast<std::size_t>(std::stoull(need_value("--threads")));
                continue;
            }
            if (arg == "--parallel-variants") {
                cfg.include_parallel_variants = true;
                continue;
            }
            if (arg == "--no-vm") {
                cfg.include_vm_variants = false;
                continue;
            }
            if (arg == "--no-llvm-jit") {
                cfg.include_llvm_jit_variants = false;
                continue;
            }
            if (arg == "--seed") {
                cfg.seed = static_cast<std::uint64_t>(std::stoull(need_value("--seed")));
                continue;
            }
            if (arg == "--backend") {
                const std::string v = need_value("--backend");
                if (v == "fast") {
                    cfg.backend = benchcalc::Backend::FastMath;
                } else if (v == "std") {
                    cfg.backend = benchcalc::Backend::StdLib;
                } else {
                    throw std::runtime_error("Unknown backend: " + v + " (expected fast|std)");
                }
                continue;
            }
            if (arg == "--abs-tol") {
                cfg.abs_tolerance = std::stof(need_value("--abs-tol"));
                continue;
            }
            if (arg == "--rel-tol") {
                cfg.rel_tolerance = std::stof(need_value("--rel-tol"));
                continue;
            }
            if (arg == "--no-verify") {
                cfg.verify = false;
                continue;
            }
            if (arg == "--csv") {
                csv_path = need_value("--csv");
                continue;
            }
            if (arg == "--json") {
                json_path = need_value("--json");
                continue;
            }
            if (arg == "--dump-plan") {
                dump_plan = true;
                continue;
            }

            throw std::runtime_error("Unknown argument: " + arg);
        }

        if (cfg.expression.empty()) {
            throw std::runtime_error("--expr is required. Example: --expr \"a+b-sin(c)\"");
        }

        if (cfg.thread_count == 0) {
            throw std::runtime_error("--threads must be >= 1");
        }

        auto run = benchcalc::run_benchmark_suite(cfg);

        const std::size_t temps = run.plan.total_buffers - run.plan.variable_buffers - run.plan.constant_buffers;

        std::cout << "Expression       : " << cfg.expression << "\n";
        std::cout << "Variables        : " << run.plan.expression.variables.size() << "\n";
        std::cout << "Constants        : " << run.plan.expression.constants.size() << "\n";
        std::cout << "Plan steps       : " << run.plan.steps.size() << "\n";
        std::cout << "Buffer count     : " << run.plan.total_buffers
                  << " (vars=" << run.plan.variable_buffers
                  << ", consts=" << run.plan.constant_buffers
                  << ", temps=" << temps << ")\n";
        std::cout << "Backend          : " << (cfg.backend == benchcalc::Backend::FastMath ? "fast_math" : "std") << "\n";
        std::cout << "LLVM JIT build   : " << (benchcalc::llvm_jit_supported_by_build() ? "ENABLED" : "DISABLED") << "\n";
        std::cout << "LLVM JIT detail  : " << benchcalc::llvm_jit_build_description() << "\n";
        const bool llvm_jit_effective = cfg.include_llvm_jit_variants && benchcalc::llvm_jit_supported_by_build();
        std::cout << "Threads          : " << cfg.thread_count << "\n";
        std::cout << "VM variants      : " << (cfg.include_vm_variants ? "ON" : "OFF") << "\n";
        std::cout << "Parallel variants: " << (cfg.include_parallel_variants ? "ON" : "OFF") << "\n";
        std::cout << "LLVM JIT variants: " << (cfg.include_llvm_jit_variants ? "ON" : "OFF")
                  << " (effective=" << (llvm_jit_effective ? "ON" : "OFF") << ")\n";
        std::cout << "Tolerance        : abs=" << cfg.abs_tolerance << ", rel=" << cfg.rel_tolerance << "\n";
        std::cout << "Dataset pool     : " << cfg.dataset_pool_size << "\n";
        std::cout << "Auto iters       : " << (cfg.target_case_ms > 0.0 ? "ON" : "OFF") << "\n";
        if (cfg.target_case_ms > 0.0) {
            std::cout << "Target per case  : " << cfg.target_case_ms << " ms (max " << cfg.max_auto_iterations << " iters)\n";
        }
        std::cout << "\n";

        if (dump_plan) {
            std::cout << "[Plan] Variables:\n";
            for (std::size_t i = 0; i < run.plan.expression.variables.size(); ++i) {
                std::cout << "  v" << i << " = " << run.plan.expression.variables[i] << "\n";
            }
            std::cout << "[Plan] Constants:\n";
            for (std::size_t i = 0; i < run.plan.expression.constants.size(); ++i) {
                std::cout << "  c" << i << " = " << run.plan.expression.constants[i] << "\n";
            }
            std::cout << "[Plan] Steps:\n";
            for (std::size_t i = 0; i < run.plan.steps.size(); ++i) {
                const auto& s = run.plan.steps[i];
                std::cout << "  #" << i << " ";
                if (s.kind == benchcalc::StepKind::Copy) {
                    std::cout << "copy  b" << s.dst << " <- b" << s.src;
                } else if (s.kind == benchcalc::StepKind::Unary) {
                    std::cout << "unary b" << s.dst << " = " << benchcalc::op_to_string(s.op) << "(b" << s.dst << ")";
                } else {
                    std::cout << "binary b" << s.dst << " = b" << s.dst << " " << benchcalc::op_to_string(s.op) << " b" << s.src;
                }
                std::cout << "\n";
            }
            std::cout << "Result buffer: b" << run.plan.result_buffer << "\n\n";
        }

        std::cout << std::left
                  << std::setw(28) << "Variant"
                  << std::setw(10) << "N"
                  << std::setw(8) << "Block"
                  << std::setw(8) << "Iters"
                  << std::setw(14) << "Median(ns)"
                  << std::setw(12) << "P10(ns)"
                  << std::setw(12) << "P90(ns)"
                  << std::setw(8) << "CV(%)"
                  << std::setw(14) << "MElem/s"
                  << std::setw(14) << "MOps/s"
                  << std::setw(12) << "GB/s(est)"
                  << std::setw(8) << "Verify"
                  << "\n";

        std::cout << std::string(148, '-') << "\n";

        for (const auto& r : run.results) {
            std::cout << std::left
                      << std::setw(28) << r.variant_name
                      << std::setw(10) << r.length
                      << std::setw(8) << r.block_size
                      << std::setw(8) << r.measured_iterations
                      << std::setw(14) << std::fixed << std::setprecision(1) << r.stats.median_ns
                      << std::setw(12) << std::fixed << std::setprecision(1) << r.stats.p10_ns
                      << std::setw(12) << std::fixed << std::setprecision(1) << r.stats.p90_ns
                      << std::setw(8) << std::fixed << std::setprecision(2) << r.stats.cv_percent
                      << std::setw(14) << std::fixed << std::setprecision(2) << r.million_elements_per_sec
                      << std::setw(14) << std::fixed << std::setprecision(2) << r.million_ops_per_sec
                      << std::setw(12) << std::fixed << std::setprecision(2) << r.estimated_gb_per_sec
                      << std::setw(8) << (r.verified ? "OK" : "FAIL")
                      << "\n";
        }

        print_best_case_summary(run);
        print_baseline_speedup(run, "step-major/switch");

        if (!csv_path.empty()) {
            save_csv(csv_path, run);
            std::cout << "\nCSV saved to: " << csv_path << "\n";
        }
        if (!json_path.empty()) {
            save_json(json_path, cfg, run);
            std::cout << "JSON saved to: " << json_path << "\n";
        }

        return 0;
    }
    catch (const std::exception& ex) {
        std::cerr << "[benchcalc] ERROR: " << ex.what() << "\n";
        return 1;
    }
}


