# BenchCalcWay

一个面向**运行时输入表达式**的 C++ benchmark 框架，目标是比较“如何排布数据与调度运算”对性能的影响。

## 当前支持

- 运算符：`+ - * /`
- 一元函数：`sin`、`sqrt`
- 常量字面量：例如 `a+2*sin(c)`、`sqrt(a*0.5)+b`
- 表达式输入：例如
  - `a+b*c/d`
  - `a+b-sin(c)`
  - `sqrt(a*b)+sin(c)`
  - `a+2*sin(c)`
- 结果写回策略：
  - 二元运算默认写回第一操作数（`dst = dst op rhs`）
  - 一元运算就地更新（`dst = op(dst)`）

## 框架设计（严格基准）

### 1) 解析与计划（Plan）

1. 先把中缀表达式转换为后缀（RPN）
2. 编译成线性 `PlanStep`：`Copy / Unary / Binary`
3. 自动做可写性处理：
   - 若某步左值不能安全覆盖（比如变量后续还要被读取），会插入 `Copy` 到临时 buffer，保证语义正确

> 这使得 `a+a*a` 这类重复变量表达式也能正确执行。

### 2) 数据生成（预热阶段）

- 根据表达式变量数量自动生成对应数量向量
- 逐样本 rejection sampling，确保表达式计算结果有限（避免 `sqrt(负数)`、`/0`）
- 生成参考输出（标量 RPN 求值），用于 correctness 校验

### 3) 统一计算模型：分块

所有变体都使用块处理（chunk/block），只比较调度策略差异：

- `step-major`: 按“步骤”扫描全数组（符合你给的示例思路）
- `block-major`: 按“块”执行全部步骤（更关注局部性）

### 4) 对比的执行变体

- `step-major/switch`
- `step-major/fnptr`
- `step-major/global-void-offset`（实验：`void()` + 全局参数流起始索引）
- `step-major/global-void-signature`（实验：`void()` + 全局签名查参）
- `block-major/switch`
- `block-major/fnptr`
- `block-major/global-void-offset`（实验：`void()` + 全局参数流起始索引）
- `block-major/global-void-signature`（实验：`void()` + 全局签名查参）
- `rpn-vm/switch`（元素级解释执行基线）
- `rpn-vm/fnptr`（元素级解释 + 函数指针分发）
- 可选并行变体（`--parallel-variants --threads N`）：
  - `step-major/*/mt`
  - `block-major/*/mt`

同时支持 unary 数学后端：

- `std`（`std::sin/std::sqrt`）
- `fast_math`（使用你的 `fast_math`）

#### `global-void-*` 两个子方案的语义

为验证“统一 `void()` 声明 + 运行时取参”方案，新增两个严格可比的子方案：

1. `global-void-offset`
   - 调度器在绑定阶段计算每一步在全局参数槽位流里的 `arg_begin`；
   - 执行时函数仅通过 `arg_begin` + 自身元数（unary/binary/copy）取参数。
2. `global-void-signature`
   - 函数通过全局签名管理器，用 `signature -> arg_begin` 映射查参数区间；
   - 调度器仅维护签名映射，不直接向函数传参数索引。

公平性保证：

- 与 `switch/fnptr` 使用同一 `ExecutionPlan`、同一块大小、同一数据集池；
- 同一 warmup / measured iterations / verify 流程；
- 仅替换“参数定位与分发路径”，其余计算内核保持一致。

### 5) LLVM JIT 执行组（新增）

支持一组 LLVM ORC JIT 变体（通过 LLVM-C API）：

- `llvm-jit/scalar`
- `llvm-jit/loop`
- `llvm-jit/scalar/mt`（当启用并行变体时）
- `llvm-jit/loop/mt`（当启用并行变体时）

说明：

- JIT 在启动后对表达式编译一次，生成两类函数：
  - 标量函数：`float f(vars, idx)`
  - 循环函数：`void f(vars, out, begin, end)`
- 对一元函数 `sin/sqrt`，JIT 会按 `--backend` 选择 lower 策略：
  - `--backend std`：LLVM intrinsic (`sin/sqrt`)
  - `--backend fast`：调用 `fast_math` 的标量包装函数（保证与框架 backend 一致）

## 构建

```powershell
cmake -S . -B build_ninja -G Ninja
cmake --build build_ninja -j
```

> `fast_math` 默认 include 路径：`../Math/fast_math/include`

可通过 CMake 参数覆盖：

```powershell
cmake -S . -B build_ninja -G Ninja -DBENCHCALC_FAST_MATH_INCLUDE="E:/Project/MelosyneTest/Math/fast_math/include"
```

启用 LLVM JIT（默认已开启），并指定 LLVM 根目录：

```powershell
cmake -S . -B build_ninja -G Ninja -DBENCHCALC_ENABLE_LLVM_JIT=ON -DBENCHCALC_LLVM_ROOT="D:/PUsing/clang+llvm-22.1.0-x86_64-pc-windows-msvc"
```

## 运行

```powershell
.\build_ninja\benchcalc.exe --expr "a+b-sin(c)" --sizes 1024,8192,65536 --blocks 32,64,128 --warmup 4 --iters 20 --backend fast
```

1-10 元更复杂场景（推荐用于调度稳定性回归）：

```powershell
.\build_ninja\benchcalc.exe --arity-suite-max 10 --arity-suite-mode complex --sizes 4096,65536 --blocks 64,128,256 --target-ms 120 --repeats 3 --no-vm --no-llvm-jit
```

说明：`complex` 模式下每个元数会自动生成 3 类表达式（链式混合 / 平衡归约 / unary-dense），因此 `1..10` 共 30 条表达式。

并行变体示例：

```powershell
.\build_ninja\benchcalc.exe --expr "a+b-sin(c)" --sizes 65536 --blocks 256 --threads 4 --parallel-variants --no-vm --target-ms 200
```

可选参数：

- `--expr` 表达式（必填）
- `--sizes` 向量长度列表
- `--blocks` 块大小列表
- `--warmup` 预热轮数
- `--iters` 计时轮数
- `--target-ms` 自动估算每个 case 的迭代次数（目标耗时，毫秒）
- `--max-auto-iters` 自动估算时的上限
- `--datasets` 数据集池大小（轮换输入，降低固定缓存形态偏置）
- `--repeats` 外层重复次数（用于稳定性/置信区间分析）
- `--arity-suite-max` 生成 `1..N` 元自动场景（不填 `--expr` 也可运行）
- `--arity-suite-mode sum|complex|both`
  - `sum`：每个元数 1 条纯加法扩展表达式
  - `complex`：每个元数 3 条复杂表达式（链式/平衡树/unary-dense）
  - `both`：同时包含 `sum + complex`
- `--threads` 并行线程数（用于 `/mt` 变体）
- `--parallel-variants` 启用并行变体
- `--no-vm` 关闭 RPN VM 基线变体
- `--no-llvm-jit` 关闭 LLVM JIT 变体
- `--seed` 随机种子
- `--backend fast|std`
- `--abs-tol` / `--rel-tol` 校验容差
- `--no-verify` 关闭结果校验
- `--csv out.csv` 导出 CSV
- `--json out.json` 导出 JSON
- `--dump-plan` 打印编译后的执行计划

## 输出指标

- `Median / P10 / P90 / Mean / Min / Max / StdDev / CV%`
- `MElem/s`：百万元素每秒
- `MOps/s`：百万算子步每秒（按 plan 的 unary+binary 步估算）
- `GB/s(est)`：按步读取/写入字节数估算
- `Verify`：与参考输出比对结果
- `Best variant by case`：每个 `(N, block)` 的最快方案汇总
- `Speedup vs baseline`：相对 `step-major/switch` 的平均加速比
- `Repeat stability summary vs baseline [step-major/fnptr]`：
  - `GeoSpeed`（几何平均速度比）
  - `CI95 Low/High`（对数域 95% 置信区间）
- 启动信息中会给出 LLVM JIT 编译支持状态与描述

## 目录结构

```text
include/benchcalc/
  types.h
  expression.h
  parser.h
  plan.h
  kernels.h
  executor.h
  dataset.h
  benchmark.h
src/
  parser.cpp
  plan.cpp
  dataset.cpp
  executor.cpp
  benchmark.cpp
  main.cpp
```

## 后续可扩展（建议）

- 统一 IR + peephole 优化（`x*1`, `x+0`）
- FMA / reciprocal 变换策略（可选且可验证误差）
- 常驻线程池（减少 `/mt` 模式线程创建成本）
- CPU 亲和 / NUMA 绑核策略
- AVX2/AVX512 自定义 binary kernel
- 表达式 hash 缓存（启动时编译一次，多次复用）
- 自动回归基准（A/B 阈值告警）

## GitHub CI / Bench 工作流

仓库内置两套工作流（位于 `.github/workflows/`）：

1. `ci.yml`
   - 在 `push / pull_request / workflow_dispatch` 触发
   - 目标：多平台编译 + 快速 smoke benchmark
   - smoke 用例：
     - 核心表达式：`a+b-sin(c)`
     - arity 复杂套件：`--arity-suite-max 10 --arity-suite-mode complex`
   - 平台/架构：
     - `ubuntu-latest` (x64)
     - `windows-latest` (x64)
     - `macos-14` (arm64)
   - 产物：每个 runner 的 smoke CSV
   - 强化项：
     - 并发控制（同分支自动取消旧 CI）
     - `timeout-minutes` 防挂起
     - `GITHUB_STEP_SUMMARY` 自动输出 smoke 最优结果

2. `bench-matrix.yml`
   - 在 `workflow_dispatch`（手动）和每周定时触发
   - 目标：多平台多架构基准跑数并上传结果
   - 默认覆盖：
     - `ubuntu-latest` (x64)
     - `windows-latest` (x64)
     - `macos-14` (arm64)
   - `macos-13` (x64) 仅在每周定时或手动开启 `include_macos13=true` 时参与（避免日常排队阻塞）
   - 支持手动参数：`sizes`、`blocks`、`target_ms`、`repeats`、`run_arity_suite`、`arity_suite_max`
   - 产物：每个 runner 的 CSV/JSON benchmark 结果
   - `GITHUB_STEP_SUMMARY` 自动输出核心两条表达式 +（可选）arity 套件最优变体

> CI 默认使用 `std` backend，且关闭 LLVM JIT/fast_math，保证跨平台可重复和依赖最小化。

## CI 数据分析与总结（2026-04-24）

本节基于 GitHub Actions 实际产物（artifact CSV）进行统计，目标是验证：

1. 多平台 smoke benchmark 是否稳定通过；  
2. 本轮 CI 补强（并发控制、矩阵调整、summary 步骤）是否影响性能结果一致性。

### 分析口径

- 表达式：`a+b-sin(c)`
- 配置：`N=4096`, `block=128`, `iters=5`, `backend=std`, `--no-vm --no-llvm-jit`
- 变体：`step-major/switch`, `step-major/fnptr`, `block-major/switch`, `block-major/fnptr`
- 数据来源：
  - 上一轮稳定版 CI：run `24843906793`
  - 本轮补强后 CI：run `24844069343`

### 1) 工作流执行健康度

| Run ID | 结论 | 覆盖平台 | 总体耗时（墙钟） |
|---:|---|---|---:|
| 24843906793 | success | ubuntu-x64 / windows-x64 / macos-arm64 | 49s |
| 24844069343 | success | ubuntu-x64 / windows-x64 / macos-arm64 | 50s |

> 说明：两次 run 耗时同量级，补强没有引入明显额外 CI 时间成本。

### 2) 各平台作业时长（本轮）

| 平台作业 | 时长 |
|---|---:|
| ubuntu-latest-x64 | 24s |
| macos-14-arm64 | 22s |
| windows-latest-x64 | 44s |

> Windows 仍是最慢构建平台，符合历史观察。

### 3) 本轮 smoke 原始结果（median_ns）

| 平台 | step-major/switch | step-major/fnptr | block-major/switch | block-major/fnptr | 平台最优 |
|---|---:|---:|---:|---:|---|
| ubuntu-latest-x64 | 21423 | 19800 | **19269** | 19349 | block-major/switch |
| windows-latest-x64 | 40800 | 23400 | 22100 | **21900** | block-major/fnptr |
| macos-14-arm64 | 27458 | 18084 | 15833 | **15375** | block-major/fnptr |

### 4) 跨平台聚合（本轮）

| 变体 | median_ns 平均（3平台） | CV% 平均 | MOps/s 平均 |
|---|---:|---:|---:|
| block-major/fnptr | **18874.7** | **5.61** | **665.13** |
| block-major/switch | 19067.3 | 10.99 | 656.61 |
| step-major/fnptr | 20428.0 | 10.95 | 608.41 |
| step-major/switch | 29893.7 | 16.96 | 440.76 |

> 从“速度 + 稳定性”双维度看，`block-major/fnptr` 是本轮 smoke 的最稳健赢家。

### 5) 与上一轮对比（prev -> latest）

#### 5.1 平台级几何平均加速比（>1 代表本轮更快）

| 平台 | 几何平均加速比 |
|---|---:|
| macos-14-arm64 | **1.374x** |
| ubuntu-latest-x64 | 1.042x |
| windows-latest-x64 | 0.986x |

#### 5.2 变体级几何平均加速比（跨平台）

| 变体 | 几何平均加速比 |
|---|---:|
| block-major/fnptr | **1.304x** |
| step-major/fnptr | 1.121x |
| block-major/switch | 1.064x |
| step-major/switch | 1.020x |

> 结论：总体不存在“补强导致系统性退化”；小幅波动主要来自 runner 噪声，方向上偏正向。

### 6) 风险与结论

1. 仍存在 GitHub Actions 平台提示：`upload-artifact@v5` 在 Node24 强制模式下会显示 Node20 兼容告警（非失败）。  
2. smoke case 规模较小（4096），对微观差异较敏感，建议把趋势判断以**跨平台聚合**为主，而不是单点极值。  
3. 当前 CI 补强后结果稳定，建议继续保持：
   - `ci.yml`：3 平台快速门禁；
   - `bench-matrix.yml`：手动/定时跑更大规模矩阵并沉淀 artifact。

### 7) bench-matrix（手动触发）结果摘要

本轮已手动触发一次 bench 矩阵工作流：

- Workflow: `bench-matrix`
- Run ID: `24844464896`
- 结论：`success`
- 覆盖平台：ubuntu x64 / windows x64 / macOS arm64
- 数据条目：96
- Verify：**96 / 96 全通过**

#### 7.1 各平台分表达式最优（按 case 相对比值几何均值）

| 平台 | 表达式组 | 最优变体 | 几何均值比值（越接近 1 越好） | case 胜出分布（top） |
|---|---|---|---:|---|
| macos-14-arm64 | `a+b*c/d` | block-major/fnptr | **1.001** | block-major/fnptr(3/4) |
| macos-14-arm64 | `a+b-sin(c)` | block-major/fnptr | **1.003** | 四变体各 1 次 |
| ubuntu-latest-x64 | `a+b*c/d` | block-major/switch | **1.021** | block-major/switch(2), block-major/fnptr(2) |
| ubuntu-latest-x64 | `a+b-sin(c)` | block-major/fnptr | **1.001** | block-major/fnptr(3/4) |
| windows-latest-x64 | `a+b*c/d` | block-major/fnptr | **1.000** | block-major/fnptr(4/4) |
| windows-latest-x64 | `a+b-sin(c)` | block-major/switch | **1.000** | block-major/switch(4/4) |

#### 7.2 跨平台全局排名（按 case 相对比值几何均值）

| 表达式组 | 第1名 | 第2名 | 第3名 | 第4名 |
|---|---|---|---|---|
| `a+b*c/d` | block-major/fnptr (**1.010**) | block-major/switch (1.018) | step-major/fnptr (1.107) | step-major/switch (1.115) |
| `a+b-sin(c)` | block-major/fnptr (**1.002**) | step-major/fnptr (1.007) | block-major/switch (1.008) | step-major/switch (1.008) |

> bench 矩阵的总体趋势与 smoke 一致：**block-major + fnptr/switch** 是当前跨平台最稳健的一档实现，`step-major/switch` 更适合作为保守基线而非最优默认。

### 8) bench-matrix 深度统计（96 条记录，全量汇总）

为了把“结果可读”升级成“可审计”，这里给出统一统计口径下的全量聚合（基于 `bench_runs/` 中 3 平台 × 2 表达式 × 2 长度 × 2 block × 4 变体，共 96 条记录）。

#### 8.1 统计定义（严谨口径）

记每个 case 为 `(platform, expr, N, block)`，每个变体测得 `median_ns = t(v, case)`。

- **case 相对比值**：`r(v, case) = t(v, case) / min_u t(u, case)`  
  - `r=1` 代表该 case 最优；越接近 1 越好。
- **全局排名指标**：对所有 case 的 `r` 取几何均值（`geo_ratio`）。
- **基线加速比**：`speedup(v, case) = t(step-major/switch, case) / t(v, case)`  
  - 大于 1 代表优于基线。

#### 8.2 数据覆盖与质量

| 项目 | 结果 |
|---|---|
| 数据来源 | `bench-matrix` run `24844464896` artifact |
| 平台 | `ubuntu-latest-x64` / `windows-latest-x64` / `macos-14-arm64` |
| 表达式 | `a+b*c/d`、`a+b-sin(c)` |
| 向量长度 | `65536`、`262144` |
| block | `128`、`256` |
| 变体数 | 4（step-major/switch、step-major/fnptr、block-major/switch、block-major/fnptr） |
| 总记录数 | 96 |
| Verify 通过率 | **96/96 = 100%** |

#### 8.3 全局聚合总表（跨平台+跨表达式）

| 变体 | geo_ratio（越小越好） | median_ns 几何均值 | MOps/s 平均 | CV% 平均 | CV% P90 |
|---|---:|---:|---:|---:|---:|
| block-major/fnptr | **1.006** | **279,690.7** | **3,503.31** | 8.11 | 16.01 |
| block-major/switch | 1.013 | 281,581.0 | 3,480.09 | 7.59 | 16.32 |
| step-major/fnptr | 1.056 | 293,474.8 | 3,200.61 | 6.83 | 14.85 |
| step-major/switch | 1.060 | 294,756.3 | 3,182.12 | **5.93** | **10.08** |

解读：

- **性能第一梯队**：`block-major/fnptr`、`block-major/switch`（两者都明显优于 step-major 两种）。
- **稳定性第一梯队（CV）**：`step-major/switch` 最稳，但速度最慢；属于“保守基线”而非“性能默认值”。

#### 8.4 相对基线加速（基线=`step-major/switch`）

| 变体 | 几何平均加速比 | 算术平均加速比 | 最小 | 最大 |
|---|---:|---:|---:|---:|
| block-major/fnptr | **1.054x** | 1.055x | 0.994x | 1.172x |
| block-major/switch | 1.047x | 1.048x | 0.971x | 1.146x |
| step-major/fnptr | 1.004x | 1.005x | 0.957x | 1.039x |
| step-major/switch | 1.000x | 1.000x | 1.000x | 1.000x |

解读：

- 以几何平均看，`block-major/fnptr` 相对保守基线有约 **5.4%** 的稳定收益。
- `step-major/fnptr` 的收益非常有限，说明函数指针本身不是关键，**关键是调度顺序（block-major）**。

#### 8.5 平台维度“夺冠次数”（每平台共 8 个 case）

| 平台 | block-major/fnptr | block-major/switch | step-major/fnptr | step-major/switch |
|---|---:|---:|---:|---:|
| macos-14-arm64 | 4 | 2 | 1 | 1 |
| ubuntu-latest-x64 | 5 | 2 | 0 | 1 |
| windows-latest-x64 | 4 | 4 | 0 | 0 |

解读：

- `block-major/fnptr` 在三个平台都保持高胜率；
- Windows 上 `block-major/switch` 与 `block-major/fnptr` 并列（4:4），说明该平台上两者差距接近噪声带。

#### 8.6 block 大小敏感性（`ratio = median_ns@256 / median_ns@128`，<1 表示 256 更快）

| 变体 | ratio 几何均值 | 最小 | 最大 |
|---|---:|---:|---:|
| block-major/switch | **0.947** | 0.875 | 1.031 |
| step-major/fnptr | 0.959 | 0.875 | 1.005 |
| step-major/switch | 0.960 | 0.884 | 1.005 |
| block-major/fnptr | 0.961 | 0.884 | 1.012 |

解读：

- 在本数据范围内，`block=256` 相比 `128` 通常有 **约 4%~5%** 的收益；
- 个别 case 会反转（ratio > 1），说明 block 并非全局常量，建议保留 sweep。

#### 8.7 规模扩展性（`N=262144 / 65536`，理论线性约 4x）

| 变体 | 比值几何均值 | 最小 | 最大 |
|---|---:|---:|---:|
| step-major/fnptr | **4.058x** | 3.864x | 4.448x |
| step-major/switch | 4.068x | 3.811x | 4.449x |
| block-major/fnptr | 4.098x | 3.835x | 4.498x |
| block-major/switch | 4.124x | 3.964x | 4.473x |

解读：

- 四个变体都接近线性扩展（4x 附近），未出现异常超线性退化；
- `step-major` 两种在该组数据里略更接近理想线性，但绝对性能仍落后于 `block-major`。

#### 8.8 表达式复杂度差异（`a+b-sin(c)` 相对 `a+b*c/d`）

`mix_over_bin = median_ns(a+b-sin(c)) / median_ns(a+b*c/d)`（同平台/同 N/同 block/同变体对齐）

| 变体 | 几何均值 | 最小 | 最大 |
|---|---:|---:|---:|
| step-major/switch | **20.125x** | 16.786x | 23.811x |
| step-major/fnptr | 20.256x | 17.220x | 23.905x |
| block-major/switch | 22.044x | 17.858x | 25.839x |
| block-major/fnptr | 22.080x | 18.647x | 26.346x |

解读：

- 在 `std::sin` 参与下，`a+b-sin(c)` 的时间量级远高于纯 `+ - * /`；
- 当一元函数成本占主导时，变体差距会被压缩到很小区间（这也是 `a+b-sin(c)` 中四变体接近的根因）。

### 9) 结论（当前阶段的默认推荐）

综合速度、跨平台一致性、以及工程可维护性：

1. **默认推荐**：`block-major/fnptr`
   - 跨平台全局 `geo_ratio` 最优（1.006）；
   - 相对基线稳定有约 5% 级别收益；
   - 与你的“运行时动态排布函数”目标一致，便于后续扩展更多算子。
2. **保守备选**：`block-major/switch`
   - 性能与第一名非常接近（1.013）；
   - 在某些平台（尤其 Windows）几乎可与 `fnptr` 打平，调试可读性更好。
3. **基线用途**：`step-major/switch`
   - 不作为默认最快路径；
   - 仍建议保留作为长期回归基线（最稳、最易解释）。

### 10) 下一步（为了让结论更“硬”）

- 在 `bench-matrix` 增加更大 `N`（例如 `1M+`）与更多 block（如 `64/128/256/512`）；
- 单独做 `std` vs `fast_math` 对照矩阵，分离“调度收益”和“函数实现收益”；
- 新增 LLVM JIT 组与非 JIT 组同口径对比表（统一 warmup、统一 target-ms、统一验证开关）；
- 引入重复 run（多次 workflow_dispatch）后做置信区间（如 bootstrap CI）评估，避免一次性样本过拟合。
