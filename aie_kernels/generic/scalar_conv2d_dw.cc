// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include <aie_api/aie.hpp>

extern "C" {

// Scalar depthwise 2D convolution, single input/output channel per group.
//
// Performs: output[c] = conv2d(input[c], weights[c], padding) for each channel
// c independently, with stride=1, dilation=1 (depthwise / groups=C). Each
// channel has its own kH*kW filter (no channel mixing).
//
// All tensors are flat 1D arrays in CHW row-major layout:
//   input   — C * H * W elements         (layout: [c][h][w])
//   weights — C * kH * kW elements       (layout: [c][kh][kw])
//   output  — C * H_out * W_out elements (layout: [c][oh][ow])
//
// Output dimensions:
//   H_out = H + 2 * padding - kH + 1
//   W_out = W + 2 * padding - kW + 1
//
// Zero-padding is applied at the boundaries.
void scalar_conv2d_dw(bfloat16 *restrict input, bfloat16 *restrict weights,
                      bfloat16 *restrict output,
                      const int32_t C, const int32_t H, const int32_t W,
                      const int32_t kH, const int32_t kW,
                      const int32_t padding)
{
    event0();

    const int32_t H_out = H + 2 * padding - kH + 1;
    const int32_t W_out = W + 2 * padding - kW + 1;

    const int32_t in_channel_stride = H * W;
    const int32_t wgt_channel_stride = kH * kW;
    const int32_t out_channel_stride = H_out * W_out;

    for (int32_t c = 0; c < C; c++) {
        for (int32_t oh = 0; oh < H_out; oh++) {
            for (int32_t ow = 0; ow < W_out; ow++) {
                float sum = 0.0f;

                // Slide the kernel over the receptive field
                for (int32_t kh = 0; kh < kH; kh++) {
                    int32_t ih = oh + kh - padding;
                    if (ih < 0 || ih >= H) continue;
                    for (int32_t kw = 0; kw < kW; kw++) {
                        int32_t iw = ow + kw - padding;
                        if (iw < 0 || iw >= W) continue;

                        float inp = static_cast<float>(
                            input[c * in_channel_stride + ih * W + iw]);
                        float wgt = static_cast<float>(
                            weights[c * wgt_channel_stride + kh * kW + kw]);
                        sum += inp * wgt;
                    }
                }

                output[c * out_channel_stride + oh * W_out + ow] =
                    static_cast<bfloat16>(sum);
            }
        }
    }

    event1();
}

}
