# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Depthwise conv on one AIE column, consolidated onto the column's own native
Shim DMA budget (2 in + 2 out channels) -- no borrowing of neighbouring
columns' Shim hardware, a hard requirement for the eventual multi-column
design where every column has its own independent workload.

THE ROUND-ROBIN + MEMTILE TRICK (why this reaches the native budget)
----------------------------------------------------------------------
Output rows are NOT split into per-core contiguous bands. Instead, within one
W-slice (see below), the image is tiled into row-tiles in natural
top-to-bottom order, and tile t is processed by core (t % num_cores).
Because consecutive GLOBAL tile indices map to DRAM addresses via a SINGLE
constant stride (h_out_tile * row_in) regardless of which physical core ends
up handling them, "which core" is not an independent DMA dimension at all --
it falls out for free as index-within-a-group-of-num_cores of one linear
sweep. This lets ONE strided Shim<->MemTile transfer per W-slice (input:
split() fan-out; output: join() gather) cover every tile of every core, with
NO limit on image height (proven empirically: h=1000, w=8 and h=200, w=61
both pass in the test suite -- the latter specifically with h_out_tile
squeezed to 1 row, the worst case for tile count), using only 2 Shim channels
for the bulk data (1 input, 1 output) PER W-SLICE.

Weights use a THIRD channel, but do NOT go through the MemTile at all: a
plain ObjectFifo's `.cons()` may be called multiple times, registering one
extra physical destination tile per call -- a native one-Shim-read,
N-core-broadcast, with no host-side data duplication and no MemTile DMA
channel cost. This matters because the MemTile itself has its own (separate
from the Shim's) limited DMA channel count: input split() + output join()
alone, at num_cores=4, already uses most of it (empirically, ALSO routing
weights through a third split() on the MemTile overflows it --
'aie.tile op number of output DMA channel exceeded!'). Total: 2 Shim
channels through the MemTile + 1 direct broadcast channel = 3, inside the
column's native 2-in/2-out budget (2 in: macro_in + weights; 1 out:
macro_out).

W-TILING: A PYTHON LOOP OF WIDE SLICES, NOT A 5TH DMA DIMENSION
----------------------------------------------------------------------
"Which W-slice" cannot be folded into the same linear sweep as H/core (two
independent spatial strides don't collapse into one), and the Shim DMA
descriptor `rt.fill()`/`rt.drain()` compiles down to
(`shim_dma_single_bd_task`) only has 4 size-dimension slots, one of which is
already spent working around an unrelated bug (see below) -- so each W-slice
gets its OWN pair of Shim<->MemTile transfers (fill+drain), issued in a
Python `for ws in range(num_w_sub)` loop, each an otherwise-complete
round-robin-H transfer restricted to that column-slice's offset.

This means the number of *separate* Shim transfers scales with num_w_sub, and
the Shim's buffer-descriptor pool is finite. IMPORTANT: "compiles" is NOT
sufficient evidence here -- 7 W-slices (15 total transfers incl. the weight
fill) compiles fine but HANGS at execution (ERT_CMD_STATE_TIMEOUT, confirmed
even on a tiny 16x400x32 case, so it is not a large-image fluke); 8 slices
(17 transfers) fails to even compile ('aie.dma_bd' op Free called on BD
chain with unassigned IDs). Only 6 slices (13 transfers) is verified
end-to-end (compiles, runs, numerically correct -- 320x320x32 and a small
16x343x32 case). tiling.py's MAX_W_SLICES=6 guards this (raise, not a
silent failure).

Within that budget, tiling.py chooses each W-slice to be as WIDE as safely
possible (bounded by the L1 budget and the row_in hardware limit below), NOT
a square-ish shape -- an earlier attempt at this session used a square
heuristic (~11-wide slices) and needed 12 slices for 128x128 alone, already
exhausting the pool; the wide-slice choice needs only 3 for the same image.
Verified numerically correct (not just "compiles"): 128x128x32 (3 slices),
320x320x32 (6 slices), 64x100x32 (2 slices, halo between slices checked).

A SECOND, independent limit exists even within num_w_sub == 1 (or within one
W-slice): the innermost DMA dimension (row_in = wp_padded * c_padded, or
w_in_tile * c_padded per-slice when W-tiled) must stay <= 2047. Exactly 2048
fails ('aie.dma_bd' op Size 0 exceeds the [0:1023] range -- almost certainly
an 11-bit unsigned field silently wrapping to 0 at 2048, hence the confusing
"Size 0"); 2047 and below work. Verified at the boundary for c=32 (61x61 ok,
62x62 fails), c=64 (29x29 ok, 30x30 fails) and c=128 (13x13 ok, 14x14 fails).
tiling.py's MAX_ROW_IN=2047 guards this too.

THE shim_dma_single_bd_task REPEAT_COUNT BUG (why every tap has a leading 1)
----------------------------------------------------------------------
`shim_dma_single_bd_task` (which every `rt.fill()`/`rt.drain()` call compiles
down to) derives `repeat_count = sizes[0] - 1` from the TAP's outermost
dimension AND separately passes the same `sizes` unchanged into the
underlying `aie.dma_bd`'s own multi-dimensional addressing -- i.e. sizes[0]
is used TWICE. If sizes[0] is a real content dimension (> 1), the resulting
task re-issues the whole (already complete) descriptor `sizes[0]` extra
times, which silently double/multi-counts a `join()` gather's expected
production events and deadlocks it (confirmed empirically this session,
root-caused by inspecting the generated lock/bd MLIR). The fix used
throughout this file: every TAP passed to `rt.fill()`/`rt.drain()` has an
explicit leading dimension of size 1 (`sizes=[1, ...]`, `strides=[0, ...]`),
which keeps `repeat_count == 0` and makes the real content dimensions start
at index 1. This consumes one of the 4 available size-dimension slots (part
of why W can't also be a DMA dimension -- see above).
"""

from ml_dtypes import bfloat16
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.helpers.taplib.tap import TensorAccessPattern

from iron.operators.vector_conv2d_dw_multicore.tiling import compute_tiling, MAX_W_SLICES


def _check_hw_limits(g, col, w, c):
    """Raise a clear ValueError if column `col`'s tiling geometry `g`
    violates a known hardware limit (see design.py module docstring).

    Must be called BEFORE `rt.sequence()`/`rt.task_group()` are opened: if a
    ValueError is raised from inside an open task group, the `with
    rt.sequence(...)` block's own cleanup masks it with an unrelated
    "Failed to close task groups" error instead of this message.
    """
    if g["num_w_sub"] > MAX_W_SLICES:
        raise ValueError(
            f"conv2d_dw_multicore: col={col}: w={w} (c={c}) needs "
            f"{g['num_w_sub']} W-slices, more than the Shim buffer-"
            f"descriptor pool can sustain (MAX_W_SLICES={MAX_W_SLICES}, "
            f"see design.py module docstring). Reduce w or c."
        )

    # The innermost DMA size (per W-slice: w_in_tile * c_padded; this equals
    # row_in only when num_w_sub == 1) must stay <= 2047 -- NOT row_in itself,
    # which is the (unbounded, stride-only) full-image row length once W is
    # tiled. See module docstring.
    row_in_tile = g["w_in_tile"] * g["c_padded"]
    if row_in_tile > 2047:
        raise ValueError(
            f"conv2d_dw_multicore: col={col}: w={w} (c={c}) gives an "
            f"innermost DMA size of {row_in_tile} (= w_in_tile "
            f"{g['w_in_tile']} * c_padded {g['c_padded']}) > 2047, the "
            f"hardware's innermost-DMA-dimension limit (see design.py "
            f"module docstring). Reduce w or c."
        )


def _make_conv_kernel(g, k_h, k_w):
    """Construct the ONE `Kernel` object to be shared by every column's
    Workers -- see `_build_column`'s docstring for why it must not be
    constructed per column."""
    c_padded = g["c_padded"]
    tile_in_elems = g["h_in_tile"] * g["w_in_tile"] * c_padded
    tile_out_elems = g["h_out_tile"] * g["w_out_tile"] * c_padded
    w_elems = k_h * k_w * c_padded
    tile_in_ty = np.ndarray[(tile_in_elems,), np.dtype[bfloat16]]
    tile_w_ty = np.ndarray[(w_elems,), np.dtype[bfloat16]]
    tile_out_ty = np.ndarray[(tile_out_elems,), np.dtype[bfloat16]]
    return Kernel(
        "vector_conv2d_dw_mc",
        "vector_conv2d_dw_mc.o",
        [tile_in_ty, tile_w_ty, tile_out_ty,
         np.int32, np.int32, np.int32, np.int32, np.int32],  # W, C, kH, kW, H_out_tile
    )


def _build_column(rt, tg, col, h, w, c, k_h, k_w, padding, num_cores, conv_fn,
                   in_tensor, w_tensor, out_tensor, num_columns=1,
                   in_base_offset=0, w_base_offset=0, out_base_offset=0):
    """Add one column's worth of ObjectFifos/Workers/DMA fill-drain ops to an
    already-open `rt`/`tg`.

    Every tile is EXPLICITLY pinned to column `col` -- MemTile: Tile(col, 1),
    Workers: Tile(col, 2..5), Shim fill/drain: Tile(col, 0). This is required
    (not just a nicety) once more than one column shares a Program:
    `SequentialPlacer` places Workers via a flat column-major sequential
    counter that ignores ObjectFifo connectivity entirely, so unpinned
    Workers would not reliably land on the same column as this group's
    MemTile. Separately, its "common column" heuristic for unpinned Shim
    endpoints only looks at an ObjectFifo's DIRECT endpoints -- for the
    MemTile-routed fifos here (split()/join()) that never includes the
    downstream compute tiles, so it would silently resolve every column's
    Shim traffic toward column 0 and spill from there -- exactly the
    cross-column Shim-borrowing this design must avoid (see module
    docstring). Hence Shim placement is pinned too, not left to inference.

    `num_columns`/`*_base_offset` let multiple columns share one flat DRAM
    tensor: `num_columns` independently-padded HWC blocks stacked
    back-to-back (NOT a literal interleaved [h,w,c*num_columns] tensor).
    Defaults (num_columns=1, offsets=0) reproduce the single-column layout.

    `conv_fn` must be ONE `Kernel(...)` object constructed once by the
    caller and passed to every `_build_column` call sharing a Program --
    constructing a fresh `Kernel` per column (even with identical name/
    signature) makes MLIR verification fail with "redefinition of symbol
    named 'vector_conv2d_dw_mc'", since each `Kernel()` call emits its own
    symbol declaration.
    """
    # Callers MUST have already validated via _check_hw_limits() before
    # opening rt.sequence()/rt.task_group() -- see that function's docstring
    # for why a raise from inside an open task group produces a confusing,
    # masked error instead of the clear message below.
    g = compute_tiling(h, w, c, k_h, k_w, padding, num_cores)

    c_padded = g["c_padded"]
    wp_padded = g["wp_padded"]
    W_out_pad = g["W_out_pad"]
    row_in = g["row_in"]
    row_out = g["row_out"]
    h_out_tile = g["h_out_tile"]
    h_in_tile = g["h_in_tile"]
    w_out_tile = g["w_out_tile"]
    w_in_tile = g["w_in_tile"]
    num_w_sub = g["num_w_sub"]
    num_tiles = g["num_tiles_total_padded"]  # H-tiles per W-slice
    H_out_pad = g["H_out_pad"]
    hp_padded = g["hp_padded"]

    tile_in_elems = h_in_tile * w_in_tile * c_padded
    tile_out_elems = h_out_tile * w_out_tile * c_padded
    w_elems = k_h * k_w * c_padded

    per_col_in_elems = hp_padded * wp_padded * c_padded
    per_col_out_elems = H_out_pad * W_out_pad * c_padded

    # L1 tile buffers (the fundamental building block: one H x one W-slice tile).
    tile_in_ty = np.ndarray[(tile_in_elems,), np.dtype[bfloat16]]
    tile_w_ty = np.ndarray[(w_elems,), np.dtype[bfloat16]]
    tile_out_ty = np.ndarray[(tile_out_elems,), np.dtype[bfloat16]]

    # --- Input: one Shim channel -> MemTile macro (one wave = num_cores
    # tiles) -> split() fan-out to the num_cores L1 tiles, fixed offsets.
    # Reused across every W-slice (only the runtime fill's DRAM offset moves,
    # see below) and across every H-wave within a slice. ---
    macro_in_ty = np.ndarray[(num_cores * tile_in_elems,), np.dtype[bfloat16]]
    of_macro_in = ObjectFifo(macro_in_ty, name=f"macro_in_c{col}", depth=2)
    sub_ins = of_macro_in.cons().split(
        offsets=[i * tile_in_elems for i in range(num_cores)],
        obj_types=[tile_in_ty] * num_cores,
        depths=[2] * num_cores,
        names=[f"l1_in_c{col}_{i}" for i in range(num_cores)],
        placement=Tile(col, 1),
    )

    # --- Output: num_cores L1 tiles -> join() gather (fixed offsets) ->
    # MemTile macro -> one Shim channel. Reused the same way. ---
    macro_out_ty = np.ndarray[(num_cores * tile_out_elems,), np.dtype[bfloat16]]
    of_macro_out = ObjectFifo(macro_out_ty, name=f"macro_out_c{col}", depth=2)
    sub_outs = of_macro_out.prod().join(
        offsets=[i * tile_out_elems for i in range(num_cores)],
        obj_types=[tile_out_ty] * num_cores,
        depths=[2] * num_cores,
        names=[f"l1_out_c{col}_{i}" for i in range(num_cores)],
        placement=Tile(col, 1),
    )

    # Weights: ONE Shim channel, broadcast to all num_cores cores directly
    # (native multi-consumer ObjectFifo -- each .cons() call registers another
    # physical destination for the same single incoming stream; no MemTile
    # hop, no host-side data duplication needed). Safe because it is
    # single-shot (filled once, reused by every wave AND every W-slice).
    of_w = ObjectFifo(tile_w_ty, name=f"w_c{col}", depth=1)
    sub_ws = [of_w.cons() for _ in range(num_cores)]

    num_waves = num_tiles // num_cores       # H-tiles processed by each core, per W-slice
    total_iters = num_waves * num_w_sub      # ... across every W-slice

    def make_body():
        def body(of_in_, of_w_, of_out_, kernel):
            elem_w = of_w_.acquire(1)
            for _ in range_(total_iters):  # hardware loop: identical calls
                elem_in = of_in_.acquire(1)
                elem_out = of_out_.acquire(1)
                kernel(elem_in, elem_w, elem_out, w_in_tile, c_padded, k_h, k_w, h_out_tile)
                of_in_.release(1)
                of_out_.release(1)
            of_w_.release(1)
        return body

    # Explicit per-core placement -- see function docstring for why this
    # can't be left to the auto-placer once more than one column is in play.
    workers = [
        Worker(
            make_body(),
            [sub_ins[i].cons(), sub_ws[i], sub_outs[i].prod(), conv_fn],
            placement=Tile(col, 2 + i),
        )
        for i in range(num_cores)
    ]

    rt.start(*workers)

    shim = Tile(col, 0)

    # Weights: single fill, broadcast to every core (see above). Sliced out
    # of the (possibly multi-column) flat weight tensor by an explicit TAP
    # rather than the no-tap default (which would fill with the WHOLE tensor).
    w_tap = TensorAccessPattern(
        (1, num_columns * w_elems),
        offset=w_base_offset,
        sizes=[1, 1, 1, w_elems],
        strides=[0, 0, 0, 1],
    )
    rt.fill(of_w.prod(), w_tensor, tap=w_tap, task_group=tg, placement=shim)

    # One pair of transfers per W-slice (Python loop -- see module
    # docstring for why W can't also be a DMA dimension). Each covers
    # every H-tile of every core within that slice via a SINGLE hardware-
    # repeating dimension (the round-robin trick), exactly like the
    # num_w_sub==1 case. Leading dim (size 1) neutralises the
    # repeat_count bug (see module docstring).
    for ws in range(num_w_sub):
        in_tap = TensorAccessPattern(
            (1, num_columns * per_col_in_elems),
            offset=in_base_offset + ws * w_out_tile * c_padded,
            sizes=[1, num_tiles, h_in_tile, w_in_tile * c_padded],
            strides=[0, h_out_tile * row_in, row_in, 1],
        )
        out_tap = TensorAccessPattern(
            (1, num_columns * per_col_out_elems),
            offset=out_base_offset + ws * w_out_tile * c_padded,
            sizes=[1, num_tiles, h_out_tile, w_out_tile * c_padded],
            strides=[0, h_out_tile * row_out, row_out, 1],
        )
        rt.fill(of_macro_in.prod(), in_tensor, tap=in_tap, task_group=tg, placement=shim)
        rt.drain(of_macro_out.cons(), out_tensor, tap=out_tap, task_group=tg, wait=True, placement=shim)


def conv2d_dw_multicore(dev, h, w, c, k_h, k_w, padding, num_cores):
    g = compute_tiling(h, w, c, k_h, k_w, padding, num_cores)
    _check_hw_limits(g, 0, w, c)
    c_padded = g["c_padded"]
    w_elems = k_h * k_w * c_padded

    # Host tensor types. Weights: one copy, broadcast to every core natively
    # (see below) -- no host-side duplication needed.
    tensor_in_ty = np.ndarray[(g["hp_padded"] * g["wp_padded"] * c_padded,), np.dtype[bfloat16]]
    tensor_w_ty = np.ndarray[(w_elems,), np.dtype[bfloat16]]
    tensor_out_ty = np.ndarray[(g["H_out_pad"] * g["W_out_pad"] * c_padded,), np.dtype[bfloat16]]

    conv_fn = _make_conv_kernel(g, k_h, k_w)

    rt = Runtime()
    with rt.sequence(tensor_in_ty, tensor_w_ty, tensor_out_ty) as (A, B, Y):
        tg = rt.task_group()
        _build_column(rt, tg, 0, h, w, c, k_h, k_w, padding, num_cores, conv_fn, A, B, Y)
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())
