# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.18)
project(bench_libtorch CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(Torch REQUIRED)

add_executable(bench_libtorch bench_libtorch.cpp)
target_link_libraries(bench_libtorch "${TORCH_LIBRARIES}")
