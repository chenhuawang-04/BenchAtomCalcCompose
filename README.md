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
- `block-major/switch`
- `block-major/fnptr`
- `rpn-vm/switch`（元素级解释执行基线）
- `rpn-vm/fnptr`（元素级解释 + 函数指针分发）
- 可选并行变体（`--parallel-variants --threads N`）：
  - `step-major/*/mt`
  - `block-major/*/mt`

同时支持 unary 数学后端：

- `std`（`std::sin/std::sqrt`）
- `fast_math`（使用你的 `fast_math`）

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
   - 在 `push / pull_request` 触发
   - 目标：多平台编译 + 快速 smoke benchmark
   - 平台/架构：
     - `ubuntu-latest` (x64)
     - `windows-latest` (x64)
     - `macos-13` (x64)
     - `macos-14` (arm64)
   - 产物：每个 runner 的 smoke CSV

2. `bench-matrix.yml`
   - 在 `workflow_dispatch`（手动）和每周定时触发
   - 目标：多平台多架构基准跑数并上传结果
   - 同样覆盖上述 4 组 OS/架构
   - 产物：每个 runner 的 CSV/JSON benchmark 结果

> CI 默认使用 `std` backend，且关闭 LLVM JIT/fast_math，保证跨平台可重复和依赖最小化。
