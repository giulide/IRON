// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include <aie_api/aie.hpp>
#include "../aie_kernel_utils.h"

// Depthwise 2D convolution — vectorized for AIE2P (512-bit vectors, 32×bf16).
//
// ASSUMPTION: The input buffer has already been zero-padded externally
// (e.g. by DMA or by a pre-processing step).  This kernel performs ONLY
// the sliding-window multiply-accumulate with no bounds checking.
//
// All tensors are flat 1D arrays in CHW row-major layout:
//   input   — C * Hp * Wp     elements  (Hp = H + 2*padding, Wp = W + 2*padding)
//   weights — C * kH * kW     elements
//   output  — C * Ho * Wo     elements  (Ho = Hp - kH + 1, Wo = Wp - kW + 1)
//
// The kernel expects stride=1, dilation=1, no groups.
//
extern "C" {

void depthwise_conv2d(bfloat16 *restrict input, bfloat16 *restrict weights,
                      bfloat16 *restrict output,
                      const int32_t C,
                      const int32_t Hp, const int32_t Wp,
                      const int32_t kH, const int32_t kW)
{
    event0();

    constexpr int VEC = 32;  // 512-bit / 16-bit bf16

    const int32_t Ho = Hp - kH + 1;
    const int32_t Wo = Wp - kW + 1;

    const int32_t in_ch_stride  = Hp * Wp;
    const int32_t wgt_ch_stride = kH * kW;
    const int32_t out_ch_stride = Ho * Wo;

    for (int32_t c = 0; c < C; c++) {

        const bfloat16 *restrict in_base  = input  + c * in_ch_stride;
        const bfloat16 *restrict wgt_base = weights + c * wgt_ch_stride;
        bfloat16 *restrict out_base       = output + c * out_ch_stride;

        for (int32_t oh = 0; oh < Ho; oh++) {

            int32_t ow = 0;

            // Vectorized main loop — process VEC output columns per iteration.
            // No bounds check: the padded input guarantees all loads are valid.
            for (; ow + VEC <= Wo; ow += VEC) {

                aie::accum<accfloat, VEC> acc = aie::zeros<accfloat, VEC>();

                for (int32_t kh = 0; kh < kH; kh++) {
                    const int32_t row_off = (oh + kh) * Wp;
                    for (int32_t kw = 0; kw < kW; kw++) {
                        const bfloat16 w = wgt_base[kh * kW + kw];
                        acc = aie::mac(acc,
                                       aie::load_v<VEC>(&in_base[row_off + ow + kw]),
                                       w);
                    }
                }

                aie::store_v(&out_base[oh * Wo + ow],
                             acc.to_vector<bfloat16>());
            }

            // Tail: remaining columns (< VEC)
            for (; ow < Wo; ow++) {
                aie::accum<accfloat, 1> sum = aie::zeros<accfloat, 1>();
                for (int32_t kh = 0; kh < kH; kh++) {
                    const int32_t row_off = (oh + kh) * Wp;
                    for (int32_t kw = 0; kw < kW; kw++) {
                        const bfloat16 w = wgt_base[kh * kW + kw];
                        sum = aie::mac(sum, in_base[row_off + ow + kw], w);
                    }
                }
                out_base[oh * Wo + ow] = sum.to_vector<bfloat16>()[0];
            }
        }
    }

    event1();
}

}