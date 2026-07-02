# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from iron.common.test_utils import torch_dtype_map


def generate_golden_reference(
    h_in: int, w_in: int, k_h: int, k_w: int, padding=0, dtype="bf16", seed=42
):
    torch.manual_seed(seed)
    val_range = 4
    dtype_torch = torch_dtype_map[dtype]

    # Come da vincolo: 1 Batch, 1 Input Channel, 1 Output Channel
    # PyTorch conv2d richiede tensori a 4 dimensioni: (Batch, Channel, Height, Width)
    input_shape = (1, 1, h_in, w_in)
    kernel_shape = (1, 1, k_h, k_w)

    # Generazione degli input randomici pesati
    X = torch.rand(input_shape, dtype=dtype_torch) * val_range
    Weight = torch.rand(kernel_shape, dtype=dtype_torch) * val_range

    # Generazione del Golden Output usando la conv2d nativa di PyTorch
    # Applichiamo i vincoli: stride=1, dilation=1
    Y = torch.nn.functional.conv2d(X, Weight, stride=1, padding=padding)

    return {
        "Input": X,
        "Kernel": Weight,
        "Output": Y,
    }