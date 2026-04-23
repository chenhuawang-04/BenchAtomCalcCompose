cmake_minimum_required(VERSION 3.20)
project(BenchCalcWay LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BENCHCALC_ENABLE_FAST_MATH "Enable fast_math backend for sin/sqrt" ON)
set(BENCHCALC_FAST_MATH_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../Math/fast_math/include" CACHE PATH "Path to fast_math include root")
option(BENCHCALC_ENABLE_LLVM_JIT "Enable LLVM ORC JIT variants" ON)
set(BENCHCALC_LLVM_ROOT "D:/PUsing/clang+llvm-22.1.0-x86_64-pc-windows-msvc" CACHE PATH "Path to LLVM distribution root")

add_executable(benchcalc
    src/main.cpp
    src/parser.cpp
    src/plan.cpp
    src/dataset.cpp
    src/executor.cpp
    src/benchmark.cpp
)

target_include_directories(benchcalc
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(benchcalc PRIVATE /O2 /W4 /permissive- /EHsc)
else()
    target_compile_options(benchcalc PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

if(BENCHCALC_ENABLE_FAST_MATH)
    if(EXISTS "${BENCHCALC_FAST_MATH_INCLUDE}/fast_math/trig.h")
        target_compile_definitions(benchcalc PRIVATE BENCHCALC_HAS_FAST_MATH=1)
        target_include_directories(benchcalc PRIVATE ${BENCHCALC_FAST_MATH_INCLUDE})
        message(STATUS "fast_math backend enabled: ${BENCHCALC_FAST_MATH_INCLUDE}")
    else()
        message(WARNING "BENCHCALC_ENABLE_FAST_MATH=ON but include not found at ${BENCHCALC_FAST_MATH_INCLUDE}. Falling back to std backend only.")
    endif()
endif()

set(BENCHCALC_LLVM_JIT_ENABLED FALSE)

if(BENCHCALC_ENABLE_LLVM_JIT)
    set(_BENCHCALC_LLVM_INCLUDE_DIR "${BENCHCALC_LLVM_ROOT}/include")
    set(_BENCHCALC_LLVM_C_API_LIB "${BENCHCALC_LLVM_ROOT}/lib/LLVM-C.lib")
    if(EXISTS "${_BENCHCALC_LLVM_INCLUDE_DIR}/llvm-c/LLJIT.h" AND EXISTS "${_BENCHCALC_LLVM_C_API_LIB}")
        message(STATUS "LLVM-C JIT backend enabled from: ${BENCHCALC_LLVM_ROOT}")
        target_compile_definitions(benchcalc PRIVATE BENCHCALC_HAS_LLVM_JIT=1)
        target_include_directories(benchcalc PRIVATE "${_BENCHCALC_LLVM_INCLUDE_DIR}")
        target_link_libraries(benchcalc PRIVATE "${_BENCHCALC_LLVM_C_API_LIB}")

        set(BENCHCALC_LLVM_JIT_ENABLED TRUE)
    else()
        message(WARNING "LLVM JIT requested but required files are missing under ${BENCHCALC_LLVM_ROOT}. Need include/llvm-c/LLJIT.h and lib/LLVM-C.lib. Building without LLVM JIT.")
    endif()
endif()

if(BENCHCALC_LLVM_JIT_ENABLED)
    target_sources(benchcalc PRIVATE src/llvm_jit.cpp)
else()
    target_sources(benchcalc PRIVATE src/llvm_jit_stub.cpp)
endif()
