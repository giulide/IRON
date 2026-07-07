#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from iron.operators.conv2d_dw.op import Conv2DDW
from iron.operators.conv2d_dw.reference import generate_golden_reference
from iron.common.test_utils import run_test


def get_params():
    # (c, h, w, k_h, k_w, padding)
    test_cases = [
        (4, 8, 8, 3, 3, 0),
        (4, 8, 8, 3, 3, 1),
        (4, 8, 8, 1, 1, 0),
    ]
    params = []
    for c, h, w, k_h, k_w, padding in test_cases:
        is_extensive = not (c == 4 and h == 8 and w == 8 and k_h == 3 and k_w == 3 and padding == 0)
        marks = [pytest.mark.extensive] if is_extensive else []
        params.append(pytest.param(c, h, w, k_h, k_w, padding, marks=marks))
    return params


def pad_input_bf16(tensor, padding):
    """Pad a (C, H, W) bf16 tensor to (C, H+2p, W+2p) with zeros."""
    if padding == 0:
        return tensor
    C, H, W = tensor.shape
    padded = torch.zeros((C, H + 2 * padding, W + 2 * padding), dtype=tensor.dtype)
    padded[:, padding:H + padding, padding:W + padding] = tensor
    return padded


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "c,h,w,k_h,k_w,padding",
    get_params(),
)
def test_conv2d_dw(c, h, w, k_h, k_w, padding, aie_context):
    golden_ref = generate_golden_reference(c, h, w, k_h, k_w, padding=padding)

    # Input must be padded externally before passing to the operator
    padded_input = pad_input_bf16(golden_ref["Input"], padding)

    # Operator works on padded dimensions: hp = h + 2*p, wp = w + 2*p
    operator = Conv2DDW(
        c=c,
        h=h + 2 * padding,
        w=w + 2 * padding,
        k_h=k_h,
        k_w=k_w,
        padding=0,  # padding handled externally
        context=aie_context,
    )

    input_buffers = {
        "input": padded_input,
        "weights": golden_ref["Kernel"],
    }
    output_buffers = {"output": golden_ref["Output"]}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-6
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    assert not errors, f"Test failed with errors: {errors}"