#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from iron.operators.scalar_conv2d.op import ScalarConv2D
from iron.operators.scalar_conv2d.reference import generate_golden_reference
from iron.common.test_utils import run_test


def get_params():
    # Small test cases for a single-column (untiled) conv2d
    test_cases = [
        # (h, w, k_h, k_w, padding)
        (4, 4, 3, 3, 0),
        (4, 4, 3, 3, 1),
        (6, 6, 3, 3, 0),
        (6, 6, 5, 5, 0),
    ]
    params = []
    for h, w, k_h, k_w, padding in test_cases:
        is_extensive = not (h == 4 and w == 4 and k_h == 3 and k_w == 3)
        marks = [pytest.mark.extensive] if is_extensive else []
        params.append(pytest.param(h, w, k_h, k_w, padding, marks=marks))
    return params


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "h,w,k_h,k_w,padding",
    get_params(),
)
def test_conv2d(h, w, k_h, k_w, padding, aie_context):
    golden_ref = generate_golden_reference(h, w, k_h, k_w, padding=padding)

    operator = ScalarConv2D(
        h=h,
        w=w,
        k_h=k_h,
        k_w=k_w,
        padding=padding,
        context=aie_context,
    )

    # Zero-pad weights to the DMA-aligned buffer size the design expects
    # (see ScalarConv2D.weight_size); the kernel never reads the padding.
    weights = golden_ref["Kernel"].reshape(-1)
    pad = operator.weight_size - weights.numel()
    if pad:
        weights = torch.nn.functional.pad(weights, (0, pad))

    input_buffers = {
        "input": golden_ref["Input"],
        "weights": weights,
    }
    output_buffers = {"output": golden_ref["Output"]}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-6
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    assert not errors, f"Test failed with errors: {errors}"
