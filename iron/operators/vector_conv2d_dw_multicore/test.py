#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from iron.operators.vector_conv2d_dw_multicore.op import VectorConv2DDWMC
from iron.operators.vector_conv2d_dw_multicore.reference import generate_golden_reference
from iron.operators.vector_conv2d_dw_multicore.tiling import MAX_W_SLICES
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
    input), so the reference covers the padded H_out_pad x W_out_pad output
    including the phantom rows/cols.
    """
    cp = x_padded.shape[-1]
    x_nchw = x_padded.permute(2, 0, 1).unsqueeze(0).contiguous()
    w_nchw = weight_hwc.permute(2, 0, 1).unsqueeze(1).contiguous()
    y = torch.nn.functional.conv2d(x_nchw, w_nchw, stride=1, padding=0, groups=cp)
    return y.squeeze(0).permute(1, 2, 0).contiguous()


def get_params():
    # (h, w, c, k_h, k_w, padding) — c=32/64/96. Narrow/medium cases need no
    # W-tiling (num_w_sub == 1): the round-robin + MemTile consolidation (see
    # design.py) applies directly, using only the column's native Shim budget
    # (1 input + 1 weight-broadcast + 1 output channel), with NO limit on
    # image height (h=1000 / h=200+w=61 below exercise this directly). Larger
    # cases (128x128 and up) need W-tiling too (num_w_sub > 1): each W-slice
    # reuses the same round-robin-H trick, issued as its own Shim<->MemTile
    # transfer pair in a Python loop -- safe up to MAX_W_SLICES slices (see
    # tiling.py / design.py module docstring for the empirically-found bound).
    test_cases = [
        (8, 8, 32, 3, 3, 1),
        (8, 16, 32, 3, 3, 1),
        (16, 8, 32, 3, 3, 1),
        (16, 16, 32, 3, 3, 1),
        (24, 24, 32, 3, 3, 1),
        (32, 32, 32, 3, 3, 1),
        (48, 48, 32, 3, 3, 1),
        (16, 48, 32, 3, 3, 1),
        (48, 16, 32, 3, 3, 1),
        (24, 40, 32, 3, 3, 1),
        (64, 16, 32, 3, 3, 1),
        (16, 16, 64, 3, 3, 1),   # c > 32: kernel loops 2 channel-groups internally
        (16, 16, 96, 3, 3, 1),   # c > 32: kernel loops 3 channel-groups internally
        (1000, 8, 32, 3, 3, 1),  # very tall: proves no height limit (round-robin core assignment)
        (61, 61, 32, 3, 3, 1),   # largest safe c=32 square with num_w_sub==1 (row_in=2016)
        (200, 61, 32, 3, 3, 1),  # worst case: w at its max (h_out_tile=1) AND tall -- proves height is
                                 # still unbounded even when every H row is its own tile (num_tiles=200)
        (64, 100, 32, 3, 3, 1),  # smallest W-tiled case: num_w_sub=2, exercises halo between W-slices
        (128, 128, 32, 3, 3, 1),  # W-tiled: num_w_sub=3
        (160, 160, 32, 3, 3, 1),  # W-tiled: num_w_sub=3
        (256, 256, 32, 3, 3, 1),  # W-tiled: num_w_sub=5
        (320, 320, 32, 3, 3, 1),  # W-tiled: num_w_sub=6, at MAX_W_SLICES
        (32, 32, 64, 3, 3, 1),   # W-tiled at higher c: num_w_sub=2
        (48, 48, 96, 3, 3, 1),   # W-tiled at higher c: num_w_sub=3
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
    x_padded = build_padded_input(x_cpadded, g["hp_padded"], g["wp_padded"], cp, padding)

    # Reference over the padded input (matches the kernel's pure conv, incl. the
    # phantom rows/cols padded up to whole sub-tiles).
    expected = depthwise_valid(x_padded, weight_cpadded)       # (H_out_pad, W_out_pad, cp)

    input_buffers = {
        "input": x_padded,
        "weights": weight_cpadded.reshape(-1),
    }
    output_buffers = {"output": expected}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-6,
        warmup_iters=10, timed_iters=50,
    )

    ns_per_elem = latency_us * 1e3 / (h * w * c)
    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"BENCH {h}x{w}x{c}: latency_us={latency_us:.2f} ns_per_elem={ns_per_elem:.4f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")

    assert not errors, f"Test failed with errors: {errors}"


@pytest.mark.supported_devices("npu2")
@pytest.mark.parametrize(
    "w,expected_num_w_sub,should_pass",
    [
        (366, MAX_W_SLICES, True),        # c=32: num_w_sub == MAX_W_SLICES (6), must compile
        (367, MAX_W_SLICES + 1, False),   # c=32: num_w_sub == MAX_W_SLICES+1 (7), must raise
    ],
)
def test_conv2d_dw_multicore_max_w_slices_guard(w, expected_num_w_sub, should_pass, aie_context):
    """MAX_W_SLICES (see tiling.py) is the empirically-found number of W-slice
    Shim<->MemTile transfer pairs that runs correctly end-to-end; one more
    slice compiles but HANGS at execution (not just a compile-time issue --
    see design.py module docstring), so it must raise a clear error instead.
    """
    operator = VectorConv2DDWMC(
        h=16, w=w, c=32, k_h=3, k_w=3, padding=1, num_cores=4, context=aie_context,
    )
    assert operator.tiling["num_w_sub"] == expected_num_w_sub
    if should_pass:
        operator.compile()  # must not raise
    else:
        with pytest.raises(ValueError, match="W-slices"):
            operator.compile()
