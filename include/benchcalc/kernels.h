#pragma once

#include "benchcalc/types.h"

#include <cstddef>

namespace benchcalc {

enum class Backend {
    StdLib,
    FastMath,
};

struct KernelConfig {
    Backend backend = Backend::FastMath;
};

using CopyKernelFn = void(*)(float* dst, const float* src, std::size_t n, const KernelConfig& cfg);
using UnaryKernelFn = void(*)(float* dst, std::size_t n, const KernelConfig& cfg);
using BinaryKernelFn = void(*)(float* dst, const float* rhs, std::size_t n, const KernelConfig& cfg);

struct KernelTable {
    CopyKernelFn copy = nullptr;

    BinaryKernelFn add = nullptr;
    BinaryKernelFn sub = nullptr;
    BinaryKernelFn mul = nullptr;
    BinaryKernelFn div = nullptr;

    UnaryKernelFn sin = nullptr;
    UnaryKernelFn sqrt = nullptr;
};

KernelTable make_kernel_table(Backend backend);

} // namespace benchcalc
