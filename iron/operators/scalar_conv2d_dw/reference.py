# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from iron.common.test_utils import torch_dtype_map


def generate_golden_reference(
    c: int, h_in: int, w_in: int, k_h: int, k_w: int, padding=0, dtype="bf16", seed=42
):
    torch.manual_seed(seed)
    val_range = 4
    dtype_torch = torch_dtype_map[dtype]

    # Depthwise conv2d (groups=C): one output channel per input channel,
    # each with its own kH*kW filter (no channel mixing).
    input_shape = (1, c, h_in, w_in)
    kernel_shape = (c, 1, k_h, k_w)

    X = torch.rand(input_shape, dtype=dtype_torch) * val_range
    Weight = torch.rand(kernel_shape, dtype=dtype_torch) * val_range

    Y = torch.nn.functional.conv2d(X, Weight, stride=1, padding=padding, groups=c)

    # Flatten to CHW layout (matching the kernel's flat 1D arrays):
    #   input:  [c][h][w]   → c * H * W + h * W + w
    #   weight: [c][kh][kw] → c * kH * kW + kh * kW + kw
    #   output: [c][oh][ow] → c * H_out * W_out + oh * W_out + ow
    return {
        "Input": X.reshape(c, h_in, w_in),  # (C, H, W)
        "Kernel": Weight.reshape(c, k_h, k_w),  # (C, kH, kW)
        "Output": Y.reshape(c, Y.shape[2], Y.shape[3]),  # (C, H_out, W_out)
    }
