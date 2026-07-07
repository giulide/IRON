// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include <aie_api/aie.hpp>

extern "C" {

// Scalar 2D convolution with a single input/output channel.
//
// Performs: output = conv2d(input, weights, padding)
// with stride=1, dilation=1, no groups.
//
// All tensors are flat 1D arrays in row-major order:
//   input   — H * W elements
//   weights — kH * kW elements
//   output  — H_out * W_out elements
//
// Output dimensions:
//   H_out = H + 2 * padding - kH + 1
//   W_out = W + 2 * padding - kW + 1
//
// Zero-padding is applied at the boundaries.
void scalar_conv2d(bfloat16 *restrict input, bfloat16 *restrict weights,
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

            // Slide the kernel over the receptive field
            for (int32_t kh = 0; kh < kH; kh++) {
                for (int32_t kw = 0; kw < kW; kw++) {
                    // Map output (oh, ow) back to input coordinates
                    int32_t ih = oh + kh - padding;
                    int32_t iw = ow + kw - padding;

                    // Zero-padding: skip out-of-bounds positions
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        float inp = static_cast<float>(input[ih * W + iw]);
                        float wgt = static_cast<float>(weights[kh * kW + kw]);
                        sum += inp * wgt;
                    }
                }
            }

            output[oh * W_out + ow] = static_cast<bfloat16>(sum);
        }
    }

    event1();
}

}