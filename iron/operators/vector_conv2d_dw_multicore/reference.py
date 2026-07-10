# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from iron.common.test_utils import torch_dtype_map


def generate_golden_reference(
    h_in: int, w_in: int, c: int, k_h: int, k_w: int, padding=0, dtype="bf16", seed=42
):
    torch.manual_seed(seed)
    val_range = 4
    dtype_torch = torch_dtype_map[dtype]

    # Depthwise conv2d (groups=C), one filter per channel. Data is generated
    # directly in HWC layout on the host; PyTorch conv2d needs NCHW, so
    # convert internally only for the reference computation.
    X_hwc = torch.rand((h_in, w_in, c), dtype=dtype_torch) * val_range
    Weight_hwc = torch.rand((k_h, k_w, c), dtype=dtype_torch) * val_range

    X_nchw = X_hwc.permute(2, 0, 1).unsqueeze(0).contiguous()  # (1, C, H, W)
    Weight_nchw = (
        Weight_hwc.permute(2, 0, 1).unsqueeze(1).contiguous()
    )  # (C, 1, kH, kW)

    Y_nchw = torch.nn.functional.conv2d(
        X_nchw, Weight_nchw, stride=1, padding=padding, groups=c
    )
    Y_hwc = Y_nchw.squeeze(0).permute(1, 2, 0).contiguous()  # (H_out, W_out, C)

    # Flat HWC layout (matching the kernel's flat 1D arrays):
    #   input:  [h][w][c]   → h * W * C + w * C + c
    #   weight: [kh][kw][c] → (kh * kW + kw) * C + c
    #   output: [oh][ow][c] → (oh * W_out + ow) * C + c
    return {
        "Input": X_hwc,  # (H, W, C)
        "Kernel": Weight_hwc,  # (kH, kW, C)
        "Output": Y_hwc,  # (H_out, W_out, C)
    }
