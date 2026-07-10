#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from iron.operators.vector_conv2d_dw_multicore.op import VectorConv2DDWMC
from iron.operators.vector_conv2d_dw_multicore.reference import generate_golden_reference
from iron.common.test_utils import run_test


def pad_channels(tensor, c_padded):
    """Zero-pad the last (channel) dimension of a (..., C) tensor to c_padded."""
    c = tensor.shape[-1]
    if c_padded == c:
        return tensor
    return torch.nn.functional.pad(tensor, (0, c_padded - c))


def build_padded_input(x_hwc, hp_padded, wp, c_padded, padding):
    """Place the (channel-padded) image inside the host-padded input buffer.

    The real image sits at [padding:padding+h, padding:padding+w]; everything
    else (the convolution border plus the extra rows needed to round the output
    up to a whole number of macro-tiles) is zero.
    """
    h, w, _ = x_hwc.shape
    out = torch.zeros((hp_padded, wp, c_padded), dtype=x_hwc.dtype)
    out[padding:padding + h, padding:padding + w, :] = x_hwc
    return out


def depthwise_valid(x_padded, weight_hwc):
    """Depthwise conv of the already-padded input with padding=0 (HWC in/out).

    Matches the multi-core kernel exactly (it is a pure conv over the pre-padded
    input), so the reference covers the padded H_out_pad x W_out output including
    the phantom rows; the valid H_out x W_out region equals the true padded conv.
    """
    cp = x_padded.shape[-1]
    x_nchw = x_padded.permute(2, 0, 1).unsqueeze(0).contiguous()
    w_nchw = weight_hwc.permute(2, 0, 1).unsqueeze(1).contiguous()
    y = torch.nn.functional.conv2d(x_nchw, w_nchw, stride=1, padding=0, groups=cp)
    return y.squeeze(0).permute(1, 2, 0).contiguous()


def get_params():
    # (h, w, c, k_h, k_w, padding) — c=32, spatial 15x15..50x50 with variable
    # widths. These exceed the single-column L1 budget: the multi-core operator
    # groups the 4 cores' L1 strips into a MemTile macro-tile that ping-pongs
    # with DRAM, so tall/wide images stream through without exhausting L1 or the
    # DMA descriptor pool.
    test_cases = [
        (16, 16, 32, 3, 3, 1),
        (24, 24, 32, 3, 3, 1),
        (32, 32, 32, 3, 3, 1),
        (48, 48, 32, 3, 3, 1),
        (16, 48, 32, 3, 3, 1),
        (48, 16, 32, 3, 3, 1),
        (24, 40, 32, 3, 3, 1),
        (64, 16, 32, 3, 3, 1),  # multi-macro streaming with row_in < 1023
        (128, 128, 32, 3, 3, 1),  # very large: stresses tiling / L1 budget
    ]
    params = []
    for h, w, c, k_h, k_w, padding in test_cases:
        is_extensive = not (h == 16 and w == 16)
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
def test_conv2d_dw_multicore(h, w, c, k_h, k_w, padding, aie_context):
    golden_ref = generate_golden_reference(h, w, c, k_h, k_w, padding=padding)

    operator = VectorConv2DDWMC(
        h=h, w=w, c=c, k_h=k_h, k_w=k_w, padding=padding,
        num_cores=4, context=aie_context,
    )
    g = operator.tiling
    cp = g["c_padded"]

    # Host-side padding: channel-pad, then place inside the padded input buffer.
    x_cpadded = pad_channels(golden_ref["Input"], cp)          # (h, w, cp)
    weight_cpadded = pad_channels(golden_ref["Kernel"], cp)    # (k_h, k_w, cp)
    x_padded = build_padded_input(x_cpadded, g["hp_padded"], g["wp"], cp, padding)

    # Reference over the padded input (matches the kernel's pure conv, incl. the
    # phantom rows padded up to a whole number of macro-tiles).
    expected = depthwise_valid(x_padded, weight_cpadded)       # (H_out_pad, W_out, cp)

    input_buffers = {
        "input": x_padded,
        "weights": weight_cpadded.reshape(-1),
    }
    output_buffers = {"output": expected}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-6
    )

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    assert not errors, f"Test failed with errors: {errors}"
