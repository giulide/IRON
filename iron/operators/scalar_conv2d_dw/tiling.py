# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tiling geometry for the H-tiled scalar depthwise conv operator (CHW layout).

Pure arithmetic (no AIE imports). Single AIE core, no vectorization: unlike
`vector_conv2d_dw_multicore/tiling.py` there is no VEC/channel-padding
concept (`c` is used exactly as given) and no W-tiling -- see this
operator's design.py module docstring for why W-tiling is intentionally
out of scope here (this design is meant for c=1, where H-tiling alone
comfortably covers image widths far beyond anything tested).

Unlike the HWC multicore design, channel is its own TensorAccessPattern
dimension here (CHW nests channel OUTSIDE row/col, so it can't be merged
into the innermost contiguous run the way HWC merges channel+col). That
means the innermost DMA dimension is `h_in_tile * wp` (one tile's
per-channel row*col-merged run), NOT `wp * c` -- the hardware 11-bit
Size-field limit (`MAX_ROW_IN`) applies to THAT product, independent of
`c`. Since every H-tile is covered by ONE strided transfer (repeat
dimension = num_h_tiles), there is no per-tile transfer-count limit
analogous to `MAX_W_SLICES` here -- the whole image, however tall, is one
Shim<->L1 transfer.
"""

# Max elements per L1 tile buffer (matches the equivalent HWC budget --
# fits the 64KB L1 once double-buffered next to the output + weight buffers).
MAX_TILE_ELEMENTS = 6144

# Hardware limit on the innermost DMA transfer dimension (elements): an
# 11-bit unsigned field that silently wraps to 0 at exactly 2048 -- see
# vector_conv2d_dw_multicore/tiling.py's MAX_ROW_IN for the empirical
# discovery. Same hardware, same limit. Here it bounds `h_in_tile * wp`
# (see module docstring), not `wp * c`.
MAX_ROW_IN = 2047


def compute_tiling(h, w, c, k_h, k_w, padding):
    """Return the H-tiled geometry (elements / rows) as a dict.

    `h`, `w` are the RAW (unpadded) image dimensions -- matches
    `vector_conv2d_dw_multicore.tiling.compute_tiling`'s convention.
    """
    wp = w + 2 * padding  # host-padded input width (conv border, never tiled)
    W_out = wp - k_w + 1
    H_out = h + 2 * padding - k_h + 1

    # Largest per-tile height fitting BOTH the L1 budget (C * h_in_tile * wp)
    # AND the hardware innermost-dimension limit (h_in_tile * wp), whichever
    # binds first.
    h_in_cap_l1 = max(k_h, MAX_TILE_ELEMENTS // (c * wp))
    h_in_cap_hw = max(k_h, MAX_ROW_IN // wp)
    h_in_cap = min(h_in_cap_l1, h_in_cap_hw)
    h_out_tile = max(1, h_in_cap - (k_h - 1))
    h_in_tile = h_out_tile + (k_h - 1)

    # The Shim DMA requires every transfer's innermost dimension to be a
    # multiple of 4 bytes (2 bf16 elements) -- empirically hit this session
    # ('aie.dma_bd' op "Transfer sizes must be multiples of 4 bytes").
    # Bump h_out_tile by 1 (which bumps h_in_tile by the same amount) until
    # BOTH the fill (h_in_tile*wp) and drain (h_out_tile*W_out) innermost
    # sizes are even; bounded loop since each step flips both parities by a
    # fixed (wp, W_out) amount, so it converges in at most a couple of steps.
    for _ in range(4):
        if (h_in_tile * wp) % 2 == 0 and (h_out_tile * W_out) % 2 == 0:
            break
        h_out_tile += 1
        h_in_tile += 1
    else:
        # Only possible when k_h is even (h_in_tile/h_out_tile then have
        # permanently opposite parities) -- not a case this operator is
        # exercised with (k_h=3 throughout this project), but fail loudly
        # rather than silently emit a misaligned transfer.
        raise ValueError(
            f"scalar_conv2d_dw: could not find a 4-byte-aligned tile size "
            f"for k_h={k_h} (even k_h can make wp/W_out parity "
            f"unsatisfiable simultaneously) -- use an odd k_h."
        )

    num_h_tiles = -(-H_out // h_out_tile)  # ceil
    H_out_pad = num_h_tiles * h_out_tile   # output rows, padded to whole tiles
    hp_padded = H_out_pad + (k_h - 1)      # padded input rows the tiles read

    return {
        "wp": wp,
        "W_out": W_out,
        "H_out": H_out,
        "h_out_tile": h_out_tile,
        "h_in_tile": h_in_tile,
        "num_h_tiles": num_h_tiles,
        "H_out_pad": H_out_pad,
        "hp_padded": hp_padded,
        # Innermost DMA dimension for both fill (h_in_tile*wp) and drain
        # (h_out_tile*W_out) -- design.py's guard checks both against
        # MAX_ROW_IN.
        "in_row_elems": h_in_tile * wp,
        "out_row_elems": h_out_tile * W_out,
    }
