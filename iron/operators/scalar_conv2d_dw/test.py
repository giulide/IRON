#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from iron.operators.scalar_conv2d_dw.op import ScalarConv2DDW
from iron.operators.scalar_conv2d_dw.reference import generate_golden_reference
from iron.common.test_utils import run_test


def get_params():
    # (c, h, w, k_h, k_w, padding) — c=32 (matches vector_conv2d_dw's VEC-aligned
    # channel count, for a direct latency/bandwidth comparison at equal problem
    # sizes), spatial size swept 1x1..14x14 (kept below the ~15x15 L1 ceiling
    # for c=32, double-buffered).
    test_cases = [(32, size, size, 3, 3, 1) for size in range(1, 15)]
    params = []
    for c, h, w, k_h, k_w, padding in test_cases:
        is_extensive = h != 8  # keep one fast case for the default (non-extensive) run
        marks = [pytest.mark.extensive] if is_extensive else []
        params.append(pytest.param(c, h, w, k_h, k_w, padding, marks=marks))
    return params


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "c,h,w,k_h,k_w,padding",
    get_params(),
)
def test_conv2d_dw_scalar(c, h, w, k_h, k_w, padding, aie_context):
    golden_ref = generate_golden_reference(c, h, w, k_h, k_w, padding=padding)

    operator = ScalarConv2DDW(
        c=c,
        h=h,
        w=w,
        k_h=k_h,
        k_w=k_w,
        padding=padding,
        context=aie_context,
    )

    # Zero-pad the per-channel weight buffer to the DMA-aligned size the
    # design expects (see ScalarConv2DDW.weight_size); the kernel never reads the padding.
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
        operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-6,
        warmup_iters=10, timed_iters=50,
    )

    ns_per_elem = latency_us * 1e3 / (h * w * c)
    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"BENCH {h}x{w}x{c}: latency_us={latency_us:.2f} ns_per_elem={ns_per_elem:.4f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    assert not errors, f"Test failed with errors: {errors}"
