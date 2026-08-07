// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include <aie_api/aie.hpp>

extern "C" {

// Scalar depthwise 2D convolution, single input/output channel per group.
//
// Performs: output[c] = conv2d(input[c], weights[c]) for each channel c
// independently, with stride=1, dilation=1, padding=0 (depthwise / groups=C).
// Each channel has its own kH*kW filter (no channel mixing).
//
// This is a PURE (valid, padding=0) convolution over an already-padded tile
// buffer -- the caller (design.py) is responsible for host-side zero-padding
// and for slicing the image into row-tiles that fit L1, mirroring
// vector_conv2d_dw_multicore's kernel: no in-kernel boundary checks, so
// tile edges never need to be told apart from true image edges here.
//
// All tensors are flat 1D arrays in CHW row-major layout:
//   input   — C * H_in * W_in elements   (layout: [c][h][w], tile-local dims)
//   weights — C * kH * kW elements       (layout: [c][kh][kw])
//   output  — C * H_out * W_out elements (layout: [c][oh][ow])
//
// Output dimensions: H_out = H_in - kH + 1, W_out = W_in - kW + 1.
void scalar_conv2d_dw(bfloat16 *restrict input, bfloat16 *restrict weights,
                      bfloat16 *restrict output,
                      const int32_t C, const int32_t H_in, const int32_t W_in,
                      const int32_t kH, const int32_t kW)
{
    event0();

    const int32_t H_out = H_in - kH + 1;
    const int32_t W_out = W_in - kW + 1;

    const int32_t in_channel_stride = H_in * W_in;
    const int32_t wgt_channel_stride = kH * kW;
    const int32_t out_channel_stride = H_out * W_out;

    for (int32_t c = 0; c < C; c++) {
        for (int32_t oh = 0; oh < H_out; oh++) {
            for (int32_t ow = 0; ow < W_out; ow++) {
                float sum = 0.0f;

                // Slide the kernel over the receptive field
                for (int32_t kh = 0; kh < kH; kh++) {
                    int32_t ih = oh + kh;
                    for (int32_t kw = 0; kw < kW; kw++) {
                        int32_t iw = ow + kw;

                        float inp = static_cast<float>(
                            input[c * in_channel_stride + ih * W_in + iw]);
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
