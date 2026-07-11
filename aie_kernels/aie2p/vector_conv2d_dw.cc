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
// SPATIAL TILING (along the H/output-row dimension only; W is never tiled):
// The design splits the output rows into strips. This kernel invocation
// produces output rows [oh_start, oh_start + H_out_tile) of the *global*
// image. The input buffer holds only the input rows this strip needs, laid
// out contiguously starting at the first *real* input row
//   ih_buf_start = max(0, oh_start - padding).
// Padding is decided against the GLOBAL image height H_global so that zero
// padding is applied only at the true top/bottom image borders, never at
// interior strip boundaries (where the neighbouring rows are real data present
// in the adjacent strip). The buffer-local input row is therefore
//   ih_local = ih_global - ih_buf_start.
// For a single strip covering the whole image (oh_start=0, H_out_tile=H_out,
// H_global=H) this reduces exactly to the untiled convolution.
//
// All tensors are flat 1D arrays in HWC row-major layout:
//   input   — (rows in buffer) * W * C elements (index: ih_local*W*C + iw*C + c)
//   weights — kH * kW * C elements              (index: (kh*kW+kw)*C + c)
//   output  — H_out_tile * W_out * C elements   (index: oh*W_out*C + ow*C + c)
//
// Full (global) output width, W untiled:
//   W_out = W + 2 * padding - kW + 1
void vector_conv2d_dw(bfloat16 *restrict input, bfloat16 *restrict weights,
                      bfloat16 *restrict output,
                      const int32_t W, const int32_t C,
                      const int32_t kH, const int32_t kW,
                      const int32_t padding,
                      const int32_t oh_start, const int32_t H_out_tile,
                      const int32_t H_global)
{
    event0();

    constexpr int32_t VEC = 32;  // 512-bit / 16-bit bf16 native lane width

    const int32_t W_out = W + 2 * padding - kW + 1;

    // First real global input row stored at buffer row 0 (see header).
    int32_t ih_buf_start = oh_start - padding;
    if (ih_buf_start < 0) ih_buf_start = 0;

    for (int32_t oh = 0; oh < H_out_tile; oh++) {
        for (int32_t ow = 0; ow < W_out; ow++) {
            // C is guaranteed a multiple of VEC by the caller: one pure
            // vectorized loop, no scalar tail needed.
            for (int32_t c = 0; c < C; c += VEC) {
                aie::accum<accfloat, VEC> acc = aie::zeros<accfloat, VEC>();

                for (int32_t kh = 0; kh < kH; kh++) {
                    // Global input row for this tap; padding decided globally.
                    int32_t ih_global = oh_start + oh + kh - padding;
                    if (ih_global < 0 || ih_global >= H_global) continue;
                    int32_t ih_local = ih_global - ih_buf_start;
                    for (int32_t kw = 0; kw < kW; kw++) {
                        int32_t iw = ow + kw - padding;
                        if (iw < 0 || iw >= W) continue;

                        const bfloat16 *in_ptr = &input[(ih_local * W + iw) * C + c];
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
