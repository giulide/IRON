// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include <aie_api/aie.hpp>

extern "C" {

// Vectorized depthwise 2D convolution, HWC layout, channel-parallel —
// MULTI-CORE variant (one column, split across the 4 cores).
//
// Same math and channel-vectorization as the single-column vector_conv2d_dw
// kernel (depthwise conv, stride 1, dilation 1, VEC=32 channels per aie::mac,
// C padded to a multiple of VEC by the caller). Two simplifications make this
// variant DMA-friendly for multi-core tiling:
//
//   1. The input is ALREADY ZERO-PADDED BY THE HOST (both H and W). So there
//      is no in-kernel padding and no boundary bounds check at all — this is a
//      pure convolution over the padded input producing the valid output. W is
//      therefore the *padded* width Wp = w + 2*padding.
//
//   2. One invocation processes a strip of H_out_tile output rows. Because the
//      input is pre-padded, output row `oh` reads input rows [oh, oh+kH) with
//      NO offset — so the strip's input buffer simply holds
//      H_out_tile + kH - 1 contiguous padded rows starting at output row 0 of
//      the strip. This regularity lets the host stream every strip of a core
//      with a single strided DMA (one buffer descriptor), instead of one DMA
//      per strip (which exhausts the Shim BD pool for tall/wide images).
//
// Flat 1D HWC row-major layout:
//   input   — (H_out_tile + kH - 1) * W * C   (index: (oh+kh)*W*C + (ow+kw)*C + c)
//   weights — kH * kW * C                      (index: (kh*kW+kw)*C + c)
//   output  — H_out_tile * W_out * C           (index: oh*W_out*C + ow*C + c)
// with W = Wp (padded width) and W_out = W - kW + 1.
void vector_conv2d_dw_mc(bfloat16 *restrict input, bfloat16 *restrict weights,
                         bfloat16 *restrict output,
                         const int32_t W, const int32_t C,
                         const int32_t kH, const int32_t kW,
                         const int32_t H_out_tile)
{
    event0();

    constexpr int32_t VEC = 32;  // 512-bit / 16-bit bf16 native lane width

    const int32_t W_out = W - kW + 1;  // input is pre-padded, so no +2*padding

    for (int32_t oh = 0; oh < H_out_tile; oh++) {
        for (int32_t ow = 0; ow < W_out; ow++) {
            // C is guaranteed a multiple of VEC by the caller: one pure
            // vectorized loop, no scalar tail needed.
            for (int32_t c = 0; c < C; c += VEC) {
                aie::accum<accfloat, VEC> acc = aie::zeros<accfloat, VEC>();

                for (int32_t kh = 0; kh < kH; kh++) {
                    for (int32_t kw = 0; kw < kW; kw++) {
                        // Pre-padded input ⇒ every tap is in bounds, no check.
                        const bfloat16 *in_ptr =
                            &input[((oh + kh) * W + (ow + kw)) * C + c];
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
