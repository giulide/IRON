// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

extern "C" {

// Naive scalar 2D convolution with a single input/output channel.
//
// Performs: output = conv2d(input, weights, padding=same/valid)
// with stride=1, dilation=1, no groups.
//
// All tensors are passed as flat 1D arrays in row-major order:
//   input  — H * W elements
//   weights — kH * kW elements
//   output — H_out * W_out elements
//
// H_out = H + 2*padding - kH + 1
// W_out = W + 2*padding - kW + 1
//
// Zero-padding is applied at the boundaries.
void conv2d_scalar(bfloat16 *restrict input, bfloat16 *restrict weights,
                   bfloat16 *restrict output,
                   const int32_t H, const int32_t W,
                   const int32_t kH, const int32_t kW,
                   const int32_t padding)
{
    event0();

    int32_t H_out = H + 2 * padding - kH + 1;
    int32_t W_out = W + 2 * padding - kW + 1;

    for (int32_t oh = 0; oh < H_out; oh++) {
        for (int32_t ow = 0; ow < W_out; ow++) {
            float sum = 0.0f;

            // Apply the kernel over the receptive field
            for (int32_t kh = 0; kh < kH; kh++) {
                for (int32_t kw = 0; kw < kW; kw++) {
                    // Map output position back to input coordinates
                    int32_t ih = oh + kh - padding;
                    int32_t iw = ow + kw - padding;

                    // Zero-padding: only accumulate if within bounds
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        float inp = aie::to_float<bfloat16>(input[ih * W + iw], 0);
                        float wgt = aie::to_float<bfloat16>(weights[kh * kW + kw], 0);
                        sum += inp * wgt;
                    }
                }
            }

            output[oh * W_out + ow] = aie::to_bfloat16(sum, 0);
        }
    }

    event1();
}

}