// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// CPU single-thread depthwise conv2d baseline, measured entirely in C++
// against LibTorch directly -- no Python interpreter anywhere in the timed
// path. Mirrors the NPU-side timing methodology in iron/common/test_utils.py
// (run_test): buffers built once, `warmup_iters` untimed calls, then
// `timed_iters` timed calls averaged (mean, not median) with a wall clock
// wrapped tightly around each call.
//
// Same grid as FINAL_BRIEF.md Section 3 (multi-column NPU real
// measurements): channel counts 32/64/128/256/512/768/1024, image sizes
// 16/64/78/114/174/366 up to each channel count's max processable width.
// Same conv parameters as the NPU design: k=3x3, padding=1, bf16,
// channels_last (HWC-equivalent), single-threaded.

#include <torch/torch.h>

#include <chrono>
#include <cstdio>
#include <vector>

namespace F = torch::nn::functional;

double run_case(int64_t c, int64_t size, int64_t k = 3, int64_t padding = 1,
                 int warmup_iters = 10, int timed_iters = 50) {
    torch::manual_seed(0);

    auto x = torch::randn({1, c, size, size}, torch::kBFloat16)
                  .contiguous(torch::MemoryFormat::ChannelsLast);
    auto weight = torch::randn({c, 1, k, k}, torch::kBFloat16);

    auto opts = F::Conv2dFuncOptions().stride(1).padding(padding).groups(c);

    for (int i = 0; i < warmup_iters; i++) {
        auto y = F::conv2d(x, weight, opts);
    }

    double total_ns = 0.0;
    for (int i = 0; i < timed_iters; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto y = F::conv2d(x, weight, opts);
        auto t1 = std::chrono::high_resolution_clock::now();
        total_ns += std::chrono::duration<double, std::nano>(t1 - t0).count();
    }
    return (total_ns / timed_iters) / 1e3;  // mean, us
}

int main() {
    at::set_num_threads(1);  // single-threaded CPU baseline, matching the
                              // established convention throughout this
                              // project: no CPU-side parallelism.

    struct Row {
        int64_t total_c;
        std::vector<int64_t> sizes;
    };
    std::vector<Row> rows = {
        {32, {16, 64, 78, 114, 174, 366}},
        {64, {16, 64, 78, 114, 174, 366}},
        {128, {16, 64, 78, 114, 174, 366}},
        {256, {16, 64, 78, 114, 174, 366}},
        {512, {16, 64, 78, 114, 174}},
        {768, {16, 64, 78, 114}},
        {1024, {16, 64, 78}},
    };

    std::printf("%8s %6s %14s\n", "channels", "size", "cpu_us");
    for (auto &row : rows) {
        for (int64_t s : row.sizes) {
            double us = run_case(row.total_c, s);
            std::printf("%8lld %6lld %14.3f\n",
                        (long long)row.total_c, (long long)s, us);
            std::fflush(stdout);
        }
    }
    return 0;
}
