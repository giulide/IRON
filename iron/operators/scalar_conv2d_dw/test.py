#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from iron.operators.scalar_conv2d_dw.op import ScalarConv2DDW
from iron.operators.scalar_conv2d_dw.reference import generate_golden_reference
from iron.common.test_utils import run_test


def build_padded_input(x_chw, hp_padded, wp, padding):
    """Place the image inside the host-padded (H and W) input buffer (CHW).

    The real image sits at [:, padding:padding+h, padding:padding+w];
    everything else (the convolution border plus the extra rows needed to
    round the output up to a whole number of H-tiles) is zero.
    """
    c, h, w = x_chw.shape
    out = torch.zeros((c, hp_padded, wp), dtype=x_chw.dtype)
    out[:, padding:padding + h, padding:padding + w] = x_chw
    return out


def depthwise_valid(x_padded, weight_chw):
    """Depthwise conv of the already-padded input with padding=0 (CHW in/out).

    Matches the kernel exactly (it is a pure conv over the pre-padded
    input), so the reference covers the padded H_out_pad x W_out output
    including the phantom rows.
    """
    c = x_padded.shape[0]
    x_nchw = x_padded.unsqueeze(0)       # (1, C, Hp, Wp)
    w_nchw = weight_chw.unsqueeze(1)     # (C, 1, kH, kW)
    y = torch.nn.functional.conv2d(x_nchw, w_nchw, stride=1, padding=0, groups=c)
    return y.squeeze(0)                  # (C, H_out_pad, W_out)


def get_params():
    # (c, h, w, k_h, k_w, padding) -- c=1 ONLY (see design.py/tiling.py
    # module docstrings: this design is intentionally scoped to c=1, where
    # H-tiling alone is sufficient and W-tiling is never needed; c>1 is not
    # exercised or supported by this test suite).
    #
    # 20x8 is a tiny forced-multi-tile sanity case (deliberately small so a
    # placement/DMA bug is cheap to catch -- same reasoning as how
    # MAX_W_SLICES was found on the HWC side with a tiny case, not a big
    # image); 88, 128, 174, 256, 366 cover the old untiled ceiling (88x88)
    # and the per-channels/column multi-column max-widths already
    # established (366 @ 32 ch/col, 174 @ 64, 114 @ 96, 78 @ 128).
    test_cases = (
        [(1, 20, 8, 3, 3, 1)] +
        [(1, size, size, 3, 3, 1) for size in (88, 128, 174, 256, 366)]
    )
    params = []
    for c, h, w, k_h, k_w, padding in test_cases:
        is_regular = h == 20 and w == 8
        marks = [] if is_regular else [pytest.mark.extensive]
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
    g = operator.tiling

    x_padded = build_padded_input(golden_ref["Input"], g["hp_padded"], g["wp"], padding)
    expected = depthwise_valid(x_padded, golden_ref["Kernel"])

    # Zero-pad the per-channel weight buffer to the DMA-aligned size the
    # design expects (see ScalarConv2DDW.weight_size); the kernel never reads the padding.
    weights = golden_ref["Kernel"].reshape(-1)
    pad = operator.weight_size - weights.numel()
    if pad:
        weights = torch.nn.functional.pad(weights, (0, pad))

    input_buffers = {
        "input": x_padded.reshape(-1),
        "weights": weights,
    }
    output_buffers = {"output": expected.reshape(-1)}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-6,
        warmup_iters=10, timed_iters=50,
    )

    ns_per_elem = latency_us * 1e3 / (h * w * c)
    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"BENCH {h}x{w}x{c}: latency_us={latency_us:.2f} ns_per_elem={ns_per_elem:.4f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    assert not errors, f"Test failed with errors: {errors}"
