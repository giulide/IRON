// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include <aie_api/aie.hpp>

extern "C" {

// Depthwise 2D convolution with multiple channels, single input/output channel per group.
//
// Performs: output[c] = conv2d(input[c], weights[c], padding)
// for each channel c independently, with stride=1, dilation=1.
//
// All tensors are flat 1D arrays in row-major order (CHW layout):
//   input   — C * H * W elements       (layout: [c][h][w])
//   weights — C * kH * kW elements     (layout: [c][kh][kw])
//   output  — C * H_out * W_out elements (layout: [c][oh][ow])
//
// Output dimensions:
//   H_out = H + 2 * padding - kH + 1
//   W_out = W + 2 * padding - kW + 1
//
// Zero-padding is applied at the boundaries.
void depthwise_conv2d(bfloat16 *restrict input, bfloat16 *restrict weights,
                      bfloat16 *restrict output,
                      const int32_t C, const int32_t H, const int32_t W,
                      const int32_t kH, const int32_t kW,
                      const int32_t padding)
{
    event0();

    int32_t H_out = H + 2 * padding - kH + 1;
    int32_t W_out = W + 2 * padding - kW + 1;

    int32_t in_channel_stride = H * W;
    int32_t wgt_channel_stride = kH * kW;
    int32_t out_channel_stride = H_out * W_out;

    for (int32_t c = 0; c < C; c++) {
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
                            float inp = aie::to_float<bfloat16>(
                                input[c * in_channel_stride + ih * W + iw], 0);
                            float wgt = aie::to_float<bfloat16>(
                                weights[c * wgt_channel_stride + kh * kW + kw], 0);
                            sum += inp * wgt;
                        }
                    }
                }

                output[c * out_channel_stride + oh * W_out + ow] = aie::to_bfloat16(sum, 0);
            }
        }
    }

    event1();
}

}