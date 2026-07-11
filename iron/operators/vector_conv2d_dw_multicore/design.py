# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ml_dtypes import bfloat16
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_
from aie.helpers.taplib.tap import TensorAccessPattern

from iron.operators.vector_conv2d_dw_multicore.tiling import compute_tiling


def conv2d_dw_multicore(dev, h, w, c, k_h, k_w, padding, num_cores):
    """Depthwise conv on one AIE column, output rows split across `num_cores`.

    Each core owns a contiguous band of output rows and streams that band as an
    L1-sized 2-D grid of sub-tiles (H and W both tiled) over time (double-
    buffered). All of a core's sub-tiles are moved with ONE strided DMA (a 4-D
    "repeat" over the sub-tile grid), so the number of DMA descriptors stays
    tiny no matter how large the image is — any size is just a finite grid of
    blocks. The input is host-pre-padded (see tiling.py), so the kernel is a
    pure conv (W = the sub-tile width) with no boundary handling and every
    sub-tile sits at a regular offset.
    """
    g = compute_tiling(h, w, c, k_h, k_w, padding, num_cores)
    c_padded = g["c_padded"]
    wp_padded = g["wp_padded"]
    W_out_pad = g["W_out_pad"]
    row_in = g["row_in"]        # = wp_padded * c_padded
    row_out = g["row_out"]      # = W_out_pad * c_padded
    h_out_tile = g["h_out_tile"]
    h_in_tile = g["h_in_tile"]
    w_out_tile = g["w_out_tile"]
    w_in_tile = g["w_in_tile"]
    num_w_sub = g["num_w_sub"]
    num_h_sub = g["num_h_sub_per_core"]
    band_rows = g["band_rows"]
    H_out_pad = g["H_out_pad"]
    hp_padded = g["hp_padded"]

    tile_in_elems = h_in_tile * w_in_tile * c_padded
    tile_out_elems = h_out_tile * w_out_tile * c_padded

    # Host tensor types (padded input, padded output — see tiling.py).
    tensor_in_ty = np.ndarray[(hp_padded * wp_padded * c_padded,), np.dtype[bfloat16]]
    tensor_w_ty = np.ndarray[(k_h * k_w * c_padded,), np.dtype[bfloat16]]
    tensor_out_ty = np.ndarray[(H_out_pad * W_out_pad * c_padded,), np.dtype[bfloat16]]

    # L1 tile buffers (the fundamental building block: a compact h_in x w_in tile).
    tile_in_ty = np.ndarray[(tile_in_elems,), np.dtype[bfloat16]]
    tile_w_ty = np.ndarray[(k_h * k_w * c_padded,), np.dtype[bfloat16]]
    tile_out_ty = np.ndarray[(tile_out_elems,), np.dtype[bfloat16]]

    # Pre-padded kernel: pure conv, W = sub-tile width, no boundary handling.
    conv_fn = Kernel(
        "vector_conv2d_dw_mc",
        "vector_conv2d_dw_mc.o",
        [tile_in_ty, tile_w_ty, tile_out_ty,
         np.int32, np.int32, np.int32, np.int32, np.int32],  # W, C, kH, kW, H_out_tile
    )

    # Per-core ObjectFifos (direct Shim <-> L1, double-buffered).
    of_in = [ObjectFifo(tile_in_ty, name=f"in{i}", depth=2) for i in range(num_cores)]
    of_w = [ObjectFifo(tile_w_ty, name=f"w{i}", depth=1) for i in range(num_cores)]
    of_out = [ObjectFifo(tile_out_ty, name=f"out{i}", depth=2) for i in range(num_cores)]

    # One worker per core: process its num_h_sub x num_w_sub sub-tiles. Every
    # kernel call is identical (pre-padded ⇒ no per-tile constants). Weights are
    # acquired once and reused across the whole band.
    num_tiles = num_h_sub * num_w_sub

    def make_body():
        def body(of_in_, of_w_, of_out_, kernel):
            elem_w = of_w_.acquire(1)
            # Hardware loop (all kernel calls identical ⇒ no unroll): keeps the
            # core program small even for hundreds of sub-tiles.
            for _ in range_(num_tiles):
                elem_in = of_in_.acquire(1)
                elem_out = of_out_.acquire(1)
                kernel(elem_in, elem_w, elem_out, w_in_tile, c_padded, k_h, k_w, h_out_tile)
                of_in_.release(1)
                of_out_.release(1)
            of_w_.release(1)
        return body

    workers = [
        Worker(
            make_body(),
            [of_in[i].cons(), of_w[i].cons(), of_out[i].prod(), conv_fn],
        )
        for i in range(num_cores)
    ]

    # Runtime: for each core, one strided DMA gathers its whole sub-tile grid in
    # (4-D: h-sub, w-sub, row-in-tile, contiguous-tile-row) and one scatters the
    # outputs. Sub-tiles overlap by (k-1) halo (stride h_out_tile < span
    # h_in_tile, stride w_out_tile < span w_in_tile).
    rt = Runtime()
    with rt.sequence(tensor_in_ty, tensor_w_ty, tensor_out_ty) as (A, B, Y):
        rt.start(*workers)
        tg = rt.task_group()
        for i in range(num_cores):
            band_start = i * band_rows  # first output row this core owns
            rt.fill(of_w[i].prod(), B, task_group=tg)
            in_tap = TensorAccessPattern(
                (1, hp_padded * wp_padded * c_padded),
                offset=band_start * row_in,
                sizes=[num_h_sub, num_w_sub, h_in_tile, w_in_tile * c_padded],
                strides=[h_out_tile * row_in, w_out_tile * c_padded, row_in, 1],
            )
            out_tap = TensorAccessPattern(
                (1, H_out_pad * W_out_pad * c_padded),
                offset=band_start * row_out,
                sizes=[num_h_sub, num_w_sub, h_out_tile, w_out_tile * c_padded],
                strides=[h_out_tile * row_out, w_out_tile * c_padded, row_out, 1],
            )
            rt.fill(of_in[i].prod(), A, tap=in_tap, task_group=tg)
            rt.drain(of_out[i].cons(), Y, tap=out_tap, task_group=tg, wait=True)
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())
