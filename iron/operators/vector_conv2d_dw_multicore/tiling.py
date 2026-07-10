# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared tiling geometry for the multi-core depthwise conv operator.

Pure arithmetic (no AIE imports) so op.py, design.py and the test agree on the
padded buffer shapes. Parallelization for one AIE column:

  * The output rows are split into `num_cores` contiguous BANDS, one per core.
  * Each core streams its band as L1-sized SUB-TILES over time (double-buffered
    DMA <-> compute), so a band larger than L1 is handled by reloading.
  * Every core's transfers are issued as ONE strided DMA (a "repeat" dimension
    over its sub-tiles) => a constant, tiny number of DMA descriptors regardless
    of image size (the per-sub-tile approach exhausts the Shim BD pool).

The host pre-pads the input (both H and W) so the kernel is a pure convolution
with no boundary handling and every sub-tile sits at a regular offset. Output
rows are padded up to a whole number of (num_cores x sub-tile) blocks; the host
slices the valid H_out x W_out region back out.
"""

import math

# Native AIE2P bf16 vector width (512-bit / 16-bit).
VEC = 32

# Max elements per L1 tile buffer (fits the 64KB L1 once double-buffered next to
# the output + weight buffers).
MAX_TILE_ELEMENTS = 6144


def compute_tiling(h, w, c, k_h, k_w, padding, num_cores):
    """Return the padded geometry (all sizes in elements / rows) as a dict."""
    c_padded = (c + VEC - 1) // VEC * VEC

    wp = w + 2 * padding          # host-padded input width
    W_out = wp - k_w + 1          # == w + 2*padding - k_w + 1 (true output width)
    row_in = wp * c_padded        # contiguous elements per padded input row
    row_out = W_out * c_padded    # contiguous elements per output row

    H_out = h + 2 * padding - k_h + 1   # true output height

    # Largest sub-tile (output rows) whose input footprint fits the L1 budget,
    # never larger than one core's fair share of the image.
    h_in_cap = max(k_h, MAX_TILE_ELEMENTS // row_in)
    h_out_tile = max(1, h_in_cap - (k_h - 1))
    h_out_tile = min(h_out_tile, max(1, math.ceil(H_out / num_cores)))
    h_in_tile = h_out_tile + (k_h - 1)

    # Distribute the sub-tiles evenly across the cores; pad up so every core runs
    # the same number (uniform buffers, one kernel signature, regular DMAs).
    num_sub_total = math.ceil(H_out / h_out_tile)
    num_sub_per_core = math.ceil(num_sub_total / num_cores)
    band_rows = num_sub_per_core * h_out_tile     # output rows per core
    H_out_pad = num_cores * band_rows             # output rows, padded
    hp_padded = H_out_pad + (k_h - 1)             # padded input rows the sub-tiles read

    return {
        "c_padded": c_padded,
        "wp": wp,
        "W_out": W_out,
        "row_in": row_in,
        "row_out": row_out,
        "H_out": H_out,
        "h_out_tile": h_out_tile,
        "h_in_tile": h_in_tile,
        "num_sub_per_core": num_sub_per_core,
        "band_rows": band_rows,
        "H_out_pad": H_out_pad,
        "hp_padded": hp_padded,
    }
