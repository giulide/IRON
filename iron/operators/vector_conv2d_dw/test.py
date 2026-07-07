#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from iron.operators.vector_conv2d_dw.op import VectorConv2DDW
from iron.operators.vector_conv2d_dw.reference import generate_golden_reference
from iron.common.test_utils import run_test


def pad_channels(tensor, c_padded):
    """Zero-pad the last (channel) dimension of a (..., C) tensor to c_padded.

    Needed for every buffer (not just weights): the compiled design DMAs
    exactly c_padded channels per pixel, so the host buffer must physically
    have that many -- even though the *values* in the padding channels only
    matter for weights (zero there forces the padded output channels to 0
    regardless of the corresponding input value).
    """
    c = tensor.shape[-1]
    if c_padded == c:
        return tensor
    return torch.nn.functional.pad(tensor, (0, c_padded - c))


def get_params():
    # (h, w, c, k_h, k_w, padding) — c=32 (exactly one VEC chunk, no channel
    # padding waste), spatial size swept 1x1..13x13 (square) to demonstrate
    # the whole point of channel-vectorization: it stays fully vectorized
    # even for tiny spatial sizes, unlike a W-vectorized design.
    test_cases = [(size, size, 32, 3, 3, 1) for size in range(1, 17)]
    params = []
    for h, w, c, k_h, k_w, padding in test_cases:
        is_extensive = h != 8  # keep one fast case for the default (non-extensive) run
        marks = [pytest.mark.extensive] if is_extensive else []
        params.append(pytest.param(h, w, c, k_h, k_w, padding, marks=marks))
    return params


@pytest.mark.supported_devices("npu2")  # AIE2P-only kernel (512-bit / VEC=32 vectors)
@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "h,w,c,k_h,k_w,padding",
    get_params(),
)
def test_conv2d_dw_vector(h, w, c, k_h, k_w, padding, aie_context):
    golden_ref = generate_golden_reference(h, w, c, k_h, k_w, padding=padding)

    operator = VectorConv2DDW(
        h=h,
        w=w,
        c=c,
        k_h=k_h,
        k_w=k_w,
        padding=padding,
        context=aie_context,
    )
    c_padded = operator.channels_padded

    # Zero-pad the channel dimension of every buffer to c_padded (see
    # VectorConv2DDW docstring / vector_conv2d_dw.cc). The extra channels
    # carry zero weight, so they contribute 0 to the (also zero-padded)
    # golden output -- no special-casing needed in the comparison.
    input_padded = pad_channels(golden_ref["Input"], c_padded)
    weights_padded = pad_channels(golden_ref["Kernel"], c_padded).reshape(-1)
    output_padded = pad_channels(golden_ref["Output"], c_padded)

    input_buffers = {
        "input": input_padded,
        "weights": weights_padded,
    }
    output_buffers = {"output": output_padded}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-6
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    assert not errors, f"Test failed with errors: {errors}"
