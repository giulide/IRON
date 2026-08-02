#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import aie.utils as aie_utils

from iron.operators.vector_conv2d_dw_multicolumn.op import VectorConv2DDWMColumn
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

    Matches the per-column kernel exactly (it is a pure conv over the
    pre-padded input), so the reference covers the padded H_out_pad x
    W_out_pad output including the phantom rows/cols.
    """
    cp = x_padded.shape[-1]
    x_nchw = x_padded.permute(2, 0, 1).unsqueeze(0).contiguous()
    w_nchw = weight_hwc.permute(2, 0, 1).unsqueeze(1).contiguous()
    y = torch.nn.functional.conv2d(x_nchw, w_nchw, stride=1, padding=0, groups=cp)
    return y.squeeze(0).permute(1, 2, 0).contiguous()


def get_params():
    # (h, w, c, k_h, k_w, padding, num_columns) -- c is PER COLUMN (total
    # channels handled = c * num_columns). num_columns=1 is a regression
    # case (must numerically match the single-column vector_conv2d_dw_multicore
    # operator). num_columns=3 specifically exercises the Shim-mis-routing
    # canary: a bug that pins Workers/MemTile per column but leaves the
    # MemTile-routed Shim fill/drain endpoints unpinned would silently try to
    # route every column's traffic through column 0's Shim budget -- this is
    # invisible at num_columns<=2 (column 0 has enough spare budget to absorb
    # a 2nd claimant by luck) and only becomes a loud, unmissable failure once
    # a 3rd column's worth of traffic oversubscribes it.
    max_columns = aie_utils.get_current_device().cols
    test_cases = [
        (16, 16, 32, 3, 3, 1, 1),   # regression: matches single-column operator
        (16, 16, 32, 3, 3, 1, 2),
        (16, 16, 64, 3, 3, 1, 2),
        (16, 16, 32, 3, 3, 1, 3),   # Shim mis-routing canary
        (16, 16, 32, 3, 3, 1, 4),
        (16, 16, 96, 3, 3, 1, 4),
        (8, 8, 32, 3, 3, 1, 8),
        (16, 16, 32, 3, 3, 1, 8),
    ]
    params = []
    for h, w, c, k_h, k_w, padding, num_columns in test_cases:
        if num_columns > max_columns:
            continue
        is_regular = num_columns == 1
        marks = [] if is_regular else [pytest.mark.extensive]
        params.append(pytest.param(h, w, c, k_h, k_w, padding, num_columns, marks=marks))
    return params


@pytest.mark.supported_devices("npu2")  # AIE2P-only kernel (512-bit / VEC=32 vectors)
@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
)
@pytest.mark.parametrize(
    "h,w,c,k_h,k_w,padding,num_columns",
    get_params(),
)
def test_conv2d_dw_multicolumn(h, w, c, k_h, k_w, padding, num_columns, aie_context):
    # One golden reference over the FULL channel range (c * num_columns):
    # depthwise (groups=C) conv is channel-separable, so slicing this into
    # num_columns chunks of c channels is mathematically identical to running
    # num_columns independent c-channel convs -- exactly what the AIE
    # program does.
    golden_ref = generate_golden_reference(h, w, c * num_columns, k_h, k_w, padding=padding)

    operator = VectorConv2DDWMColumn(
        h=h, w=w, c=c, k_h=k_h, k_w=k_w, padding=padding,
        num_cores=4, num_columns=num_columns, context=aie_context,
    )
    g = operator.tiling
    cp = g["c_padded"]

    in_blocks = []
    w_blocks = []
    out_blocks = []
    for col in range(num_columns):
        x_slice = golden_ref["Input"][:, :, col * c:(col + 1) * c]
        w_slice = golden_ref["Kernel"][:, :, col * c:(col + 1) * c]

        x_cpadded = pad_channels(x_slice, cp)           # (h, w, cp)
        weight_cpadded = pad_channels(w_slice, cp)       # (k_h, k_w, cp)
        x_padded = build_padded_input(x_cpadded, g["hp_padded"], g["wp_padded"], cp, padding)

        # Reference over the padded input (matches the kernel's pure conv,
        # incl. the phantom rows/cols padded up to whole sub-tiles).
        expected = depthwise_valid(x_padded, weight_cpadded)  # (H_out_pad, W_out_pad, cp)

        in_blocks.append(x_padded.reshape(-1))
        w_blocks.append(weight_cpadded.reshape(-1))
        out_blocks.append(expected.reshape(-1))

    # `num_columns` independently-padded HWC blocks stacked back-to-back --
    # see design.py module docstring. NOT a literal interleaved
    # [h, w, c*num_columns] tensor.
    input_buffers = {
        "input": torch.cat(in_blocks),
        "weights": torch.cat(w_blocks),
    }
    output_buffers = {"output": torch.cat(out_blocks)}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-6,
        warmup_iters=10, timed_iters=50,
    )

    total_c = c * num_columns
    ns_per_elem = latency_us * 1e3 / (h * w * total_c)
    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"BENCH {h}x{w}x{total_c} (cols={num_columns}): latency_us={latency_us:.2f} ns_per_elem={ns_per_elem:.4f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    assert not errors, f"Test failed with errors: {errors}"
