# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scalar depthwise conv (CHW layout, single AIE core), H-tiled to handle
images far beyond the single-shot L1 ceiling -- intended for use at c=1
(see tiling.py module docstring for why W-tiling is out of scope: at c=1
the hardware row-size limit alone allows widths far past anything this
operator is exercised at, so only H needs tiling).

Unlike vector_conv2d_dw_multicore's HWC design, there is no MemTile hop or
multi-core fan-out here -- one core, direct Shim<->L1<->Shim, so the whole
image (however tall) is covered by a SINGLE strided Shim transfer: channel
is its own TensorAccessPattern dimension (CHW can't merge channel into the
innermost contiguous run the way HWC merges channel+col), and because
every H-tile is folded into that same transfer's repeat dimension, there
is no per-tile transfer-count limit analogous to MAX_W_SLICES here.

The kernel (aie_kernels/generic/scalar_conv2d_dw.cc) does a PURE (padding=0)
convolution over an already-padded tile buffer -- the host pre-pads H
spatially, mirroring vector_conv2d_dw_multicore's proven pattern.
"""

from ml_dtypes import bfloat16
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_
from aie.helpers.taplib.tap import TensorAccessPattern

from iron.operators.scalar_conv2d_dw.tiling import compute_tiling, MAX_ROW_IN


def _check_hw_limits(g, h, w, c):
    """Raise a clear ValueError if the tiling geometry `g` violates the
    hardware innermost-DMA-dimension limit (see tiling.py module docstring).

    Must be called BEFORE `rt.sequence()`/`rt.task_group()` are opened: a
    ValueError raised from inside an open task group gets masked by an
    unrelated "Failed to close task groups" error instead of this message
    (see vector_conv2d_dw_multicore/design.py's `_check_hw_limits` for the
    same issue found and fixed there).
    """
    if g["in_row_elems"] > MAX_ROW_IN:
        raise ValueError(
            f"scalar_conv2d_dw: h={h} w={w} (c={c}) gives an innermost "
            f"input DMA size of {g['in_row_elems']} (= h_in_tile "
            f"{g['h_in_tile']} * wp {g['wp']}) > {MAX_ROW_IN}, the "
            f"hardware's innermost-DMA-dimension limit (see tiling.py "
            f"module docstring). Reduce w."
        )
    if g["out_row_elems"] > MAX_ROW_IN:
        raise ValueError(
            f"scalar_conv2d_dw: h={h} w={w} (c={c}) gives an innermost "
            f"output DMA size of {g['out_row_elems']} (= h_out_tile "
            f"{g['h_out_tile']} * W_out {g['W_out']}) > {MAX_ROW_IN}, the "
            f"hardware's innermost-DMA-dimension limit (see tiling.py "
            f"module docstring). Reduce w."
        )


def conv2d_scalar_dw(dev, c, h, w, k_h, k_w, padding):
    g = compute_tiling(h, w, c, k_h, k_w, padding)
    _check_hw_limits(g, h, w, c)

    wp = g["wp"]
    W_out = g["W_out"]
    h_out_tile = g["h_out_tile"]
    h_in_tile = g["h_in_tile"]
    num_h_tiles = g["num_h_tiles"]
    H_out_pad = g["H_out_pad"]
    hp_padded = g["hp_padded"]

    # Shim DMA requires transfer lengths to be a multiple of 4 elements;
    # pad the (small, single-shot) per-channel weight buffer accordingly
    # (matches the original untiled design's weight_size logic).
    w_elems = (c * k_h * k_w + 3) // 4 * 4

    tile_in_elems = c * h_in_tile * wp
    tile_out_elems = c * h_out_tile * W_out

    # Host tensor types (CHW, padded on H only -- see module docstring).
    tensor_in_ty = np.ndarray[(c * hp_padded * wp,), np.dtype[bfloat16]]
    tensor_w_ty = np.ndarray[(w_elems,), np.dtype[bfloat16]]
    tensor_out_ty = np.ndarray[(c * H_out_pad * W_out,), np.dtype[bfloat16]]

    # L1 tile buffers (one H-tile, every channel).
    tile_in_ty = np.ndarray[(tile_in_elems,), np.dtype[bfloat16]]
    tile_w_ty = np.ndarray[(w_elems,), np.dtype[bfloat16]]
    tile_out_ty = np.ndarray[(tile_out_elems,), np.dtype[bfloat16]]

    conv_fn = Kernel(
        "scalar_conv2d_dw",
        "scalar_conv2d_dw.o",
        [tile_in_ty, tile_w_ty, tile_out_ty,
         np.int32, np.int32, np.int32, np.int32, np.int32],  # C, H_in, W_in, kH, kW
    )

    of_in = ObjectFifo(tile_in_ty, name="in", depth=2)
    of_w = ObjectFifo(tile_w_ty, name="weights", depth=1)
    of_out = ObjectFifo(tile_out_ty, name="out", depth=2)

    def core_body(of_in_, of_w_, of_out_, kernel):
        elem_w = of_w_.acquire(1)
        for _ in range_(num_h_tiles):  # hardware loop: identical calls
            elem_in = of_in_.acquire(1)
            elem_out = of_out_.acquire(1)
            kernel(elem_in, elem_w, elem_out, c, h_in_tile, wp, k_h, k_w)
            of_in_.release(1)
            of_out_.release(1)
        of_w_.release(1)

    worker = Worker(core_body, [of_in.cons(), of_w.cons(), of_out.prod(), conv_fn])

    rt = Runtime()
    with rt.sequence(tensor_in_ty, tensor_w_ty, tensor_out_ty) as (A, B, Y):
        rt.start(worker)
        tg = rt.task_group()

        # Weights: single fill, single-shot (small, not tiled).
        rt.fill(of_w.prod(), B, task_group=tg)

        # ONE strided transfer covers every H-tile of every channel: channel
        # is its own dimension (CHW can't merge it into the innermost run
        # the way HWC does), row*col merge into one contiguous run per tile
        # since each tile spans the full (padded) width. Leading dim (size
        # 1) neutralises the repeat_count bug (see
        # vector_conv2d_dw_multicore/design.py's module docstring for the
        # same issue on the same hardware).
        in_tap = TensorAccessPattern(
            (1, c * hp_padded * wp),
            offset=0,
            sizes=[1, num_h_tiles, c, h_in_tile * wp],
            strides=[0, h_out_tile * wp, hp_padded * wp, 1],
        )
        out_tap = TensorAccessPattern(
            (1, c * H_out_pad * W_out),
            offset=0,
            sizes=[1, num_h_tiles, c, h_out_tile * W_out],
            strides=[0, h_out_tile * W_out, H_out_pad * W_out, 1],
        )
        rt.fill(of_in.prod(), A, tap=in_tap, task_group=tg)
        rt.drain(of_out.cons(), Y, tap=out_tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())
