# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared tiling geometry for the multi-core depthwise conv operator.

Pure arithmetic (no AIE imports) so op.py, design.py and the test agree on the
padded buffer shapes. Parallelization for one AIE column:

  * The output ROWS are split into `num_cores` contiguous BANDS, one per core.
  * Each core streams its band as an L1-sized 2-D GRID of SUB-TILES (H and W
    both tiled) over time (double-buffered DMA <-> compute). Tiling W too means
    a single wide row that would overflow L1 is handled, so ANY image size works
    — the whole matrix is just a finite grid of blocks the 4 cores chew through.
  * Every core's transfers are issued as ONE strided DMA (a 4-D "repeat" over
    its sub-tile grid) => a constant, tiny number of DMA descriptors.

The host pre-pads the input (both H and W) so the kernel is a pure convolution
with no boundary handling and every sub-tile sits at a regular offset. Output
rows/cols are padded up to whole sub-tiles; the host slices the valid
H_out x W_out region back out. For a narrow image (a k_h-row full-width band
fits L1) this reduces to full-width strips: num_w_sub == 1.
"""

import math

# Native AIE2P bf16 vector width (512-bit / 16-bit).
VEC = 32

# Max elements per L1 tile buffer (fits the 64KB L1 once double-buffered next to
# the output + weight buffers).
MAX_TILE_ELEMENTS = 6144

# Hardware limit on the innermost DMA transfer dimension (elements), found
# empirically this session: a per-slice row (w_in_tile * c_padded, or
# wp_padded * c_padded for the num_w_sub==1 case) of exactly 2048 fails
# ('aie.dma_bd' op Size 0 exceeds the [0:1023] range -- almost certainly an
# 11-bit unsigned field silently wrapping to 0), 2047 and below work.
MAX_ROW_IN = 2047

# Max number of W-slices (each a separate pair of Shim<->MemTile transfers,
# looped in Python -- see design.py) that runs correctly. IMPORTANT: this is
# a RUNTIME limit, not just a compile-time one -- 7 slices (15 total
# transfers incl. the weight fill) COMPILES fine but hangs
# (ERT_CMD_STATE_TIMEOUT) at execution; 8 slices (17 transfers) fails to
# compile at all ('aie.dma_bd' op Free called on BD chain with unassigned
# IDs). Only 6 slices (13 transfers) was verified to both compile AND run
# with a numerically correct result. Kept as a named constant so the guard
# in design.py explains itself.
MAX_W_SLICES = 6


def compute_tiling(h, w, c, k_h, k_w, padding, num_cores):
    """Return the padded 2-D tiling geometry (elements / rows / cols) as a dict."""
    c_padded = (c + VEC - 1) // VEC * VEC

    wp = w + 2 * padding                 # host-padded input width (conv border)
    W_out = wp - k_w + 1                 # true output width
    H_out = h + 2 * padding - k_h + 1    # true output height

    # --- Width tiling ---
    # Full width only when a k_h-row full-width band already fits L1 AND its
    # row (wp*c_padded) is under the hardware's innermost-dimension limit
    # (narrow image: tiling W would only add halo for nothing anyway).
    # Otherwise, tile W as WIDE as safely possible -- bounded by the L1 budget
    # (with h_in_tile at its k_h minimum) AND the MAX_ROW_IN hardware limit --
    # rather than a square-ish heuristic: each W-slice reuses the full
    # round-robin-H trick internally (see design.py), so wider slices mean
    # fewer separate Shim transfers (the actual scarce resource -- see
    # MAX_W_SLICES), not less L1 headroom.
    max_w_in_l1 = MAX_TILE_ELEMENTS // (k_h * c_padded)
    max_w_in_hw = MAX_ROW_IN // c_padded
    max_w_in = max(k_w, min(max_w_in_l1, max_w_in_hw))
    if max_w_in >= wp and wp * c_padded <= MAX_ROW_IN:
        w_in_tile = wp
        w_out_tile = W_out
        num_w_sub = 1
    else:
        max_w_out = max(1, max_w_in - (k_w - 1))
        num_w_sub = math.ceil(W_out / max_w_out)
        w_out_tile = math.ceil(W_out / num_w_sub)  # rebalance evenly
        w_in_tile = w_out_tile + (k_w - 1)
    W_out_pad = num_w_sub * w_out_tile             # output cols, padded to whole tiles
    wp_padded = W_out_pad + (k_w - 1)              # padded input cols the tiles read

    # --- Height tiling: largest sub-tile height fitting L1 given the tile width ---
    h_in_cap = max(k_h, MAX_TILE_ELEMENTS // (w_in_tile * c_padded))
    h_out_tile = max(1, h_in_cap - (k_h - 1))
    h_out_tile = min(h_out_tile, max(1, math.ceil(H_out / num_cores)))
    h_in_tile = h_out_tile + (k_h - 1)

    # Distribute the H sub-tiles evenly across the cores (uniform buffers).
    num_h_sub_total = math.ceil(H_out / h_out_tile)
    num_h_sub_per_core = math.ceil(num_h_sub_total / num_cores)
    band_rows = num_h_sub_per_core * h_out_tile    # output rows per core
    H_out_pad = num_cores * band_rows              # output rows, padded
    hp_padded = H_out_pad + (k_h - 1)              # padded input rows the tiles read

    row_in = wp_padded * c_padded                  # elements per padded input row
    row_out = W_out_pad * c_padded                 # elements per padded output row

    # Round-robin tile assignment (used only when num_w_sub == 1, i.e. no
    # W-tiling): tiles are numbered 0..num_tiles_total_padded-1 in natural
    # top-to-bottom image order and handed out core = tile_index % num_cores.
    # "Which core" is then a pure function of position within one linear
    # sweep, so a SINGLE hardware-repeating DMA dimension (stride =
    # h_out_tile*row_in) covers every tile of every core at once --
    # consolidating all cores onto one Shim channel (via split()/join()
    # through the MemTile) with NO limit on image height. This does NOT
    # extend to num_w_sub > 1: seeARCHITECTURE.md / design.py module
    # docstring for why (2 independent spatial axes + core selection exceed
    # the 3 usable descriptor dimensions once the shim_dma_single_bd_task
    # repeat_count bug's leading dummy dimension is accounted for).
    num_waves = num_h_sub_per_core          # tiles each core handles
    num_tiles_total_padded = num_cores * num_waves

    return {
        "c_padded": c_padded,
        "wp_padded": wp_padded,
        "W_out": W_out,
        "W_out_pad": W_out_pad,
        "row_in": row_in,
        "row_out": row_out,
        "H_out": H_out,
        "h_out_tile": h_out_tile,
        "h_in_tile": h_in_tile,
        "w_out_tile": w_out_tile,
        "w_in_tile": w_in_tile,
        "num_w_sub": num_w_sub,
        "num_h_sub_per_core": num_h_sub_per_core,
        "band_rows": band_rows,
        "H_out_pad": H_out_pad,
        "hp_padded": hp_padded,
        "num_waves": num_waves,
        "num_tiles_total_padded": num_tiles_total_padded,
    }
