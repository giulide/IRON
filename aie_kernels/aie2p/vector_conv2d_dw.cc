// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include <aie_api/aie.hpp>

extern "C" {

// Vectorized depthwise 2D convolution, HWC layout, channel-parallel.
//
// Performs: output[oh,ow,c] = sum_{kh,kw} input[oh+kh-pad, ow+kw-pad, c] * weights[kh,kw,c]
// with stride=1, dilation=1 (depthwise / groups=C, no channel mixing).
//
// Vectorization is across the CHANNEL dimension (not spatial position): for a
// fixed output pixel and a fixed kernel tap (kh,kw), VEC=32 channels are
// processed in one aie::mac as an element-wise vector*vector multiply-add (no
// cross-lane reduction -- each channel accumulates independently). This keeps
// the design effective even for small H/W, unlike vectorizing along W (which
// needs W_out >= VEC to do any vectorized work at all).
//
// C is expected to already be a multiple of VEC (the caller rounds the real
// channel count up and zero-pads the extra weight channels -- see
// ScalarConv2DDW-sibling op.py's `channels_padded`). This lets the whole
// kernel be one pure vectorized loop with no scalar remainder/tail path: the
// padded channels simply produce output 0 (multiplied by a zero weight) and
// are discarded by the caller.
//
// Zero-padding at spatial boundaries is handled in-kernel (bounds check per
// tap; no host-side pre-padding needed): the bounds check only depends on
// (oh,ow,kh,kw), which is identical for every channel in a lane group, so a
// single scalar branch decides whether to skip the whole vector MAC for that
// tap -- no per-lane masking required.
//
// All tensors are flat 1D arrays in HWC row-major layout:
//   input   — H * W * C elements         (index: h*W*C + w*C + c)
//   weights — kH * kW * C elements       (index: (kh*kW+kw)*C + c)
//   output  — H_out * W_out * C elements (index: oh*W_out*C + ow*C + c)
//
// Output dimensions:
//   H_out = H + 2 * padding - kH + 1
//   W_out = W + 2 * padding - kW + 1
void vector_conv2d_dw(bfloat16 *restrict input, bfloat16 *restrict weights,
                      bfloat16 *restrict output,
                      const int32_t H, const int32_t W, const int32_t C,
                      const int32_t kH, const int32_t kW,
                      const int32_t padding)
{
    event0();

    constexpr int32_t VEC = 32;  // 512-bit / 16-bit bf16 native lane width

    const int32_t H_out = H + 2 * padding - kH + 1;
    const int32_t W_out = W + 2 * padding - kW + 1;

    for (int32_t oh = 0; oh < H_out; oh++) {
        for (int32_t ow = 0; ow < W_out; ow++) {
            // C is guaranteed a multiple of VEC by the caller: one pure
            // vectorized loop, no scalar tail needed.
            for (int32_t c = 0; c < C; c += VEC) {
                aie::accum<accfloat, VEC> acc = aie::zeros<accfloat, VEC>();

                for (int32_t kh = 0; kh < kH; kh++) {
                    int32_t ih = oh + kh - padding;
                    if (ih < 0 || ih >= H) continue;
                    for (int32_t kw = 0; kw < kW; kw++) {
                        int32_t iw = ow + kw - padding;
                        if (iw < 0 || iw >= W) continue;

                        const bfloat16 *in_ptr = &input[(ih * W + iw) * C + c];
                        const bfloat16 *w_ptr = &weights[(kh * kW + kw) * C + c];
                        acc = aie::mac(acc, aie::load_v<VEC>(in_ptr),
                                       aie::load_v<VEC>(w_ptr));
                    }
                }

                aie::store_v(&output[(oh * W_out + ow) * C + c],
                             acc.to_vector<bfloat16>());
            }
        }
    }

    event1();
}

}
