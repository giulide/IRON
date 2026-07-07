# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ml_dtypes import bfloat16
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_


# Maximum elements per single-channel tile to stay within 64KB L1
# (2 buffers I/O ~6K elems × 2 bytes = 12KB, weights negligible, total ≪ 64KB)
_MAX_ELEMS = 6144
_W_TILE = 32   # fixed tile width (matches 512-bit / 16-bit bf16 vectors)
_H_TILE = _MAX_ELEMS // _W_TILE  # 192 rows max


def conv2d_dw_design(dev, c, h, w, k_h, k_w, padding, num_aie_columns):
    # Padded input dimensions (padding is applied externally, e.g. on the host)
    Hp = h + 2 * padding
    Wp = w + 2 * padding

    # Output dimensions (after convolution on padded input)
    Ho = Hp - k_h + 1
    Wo = Wp - k_w + 1

    # Number of channels per column (round-robin distribution)
    channels_per_column = (c + num_aie_columns - 1) // num_aie_columns

    # --- Spatial tiling parameters ---
    # Overlap between adjacent strips (kernel slides by kW/kH):
    #   - horizontal: output advances by _W_TILE - kW + 1 columns,
    #     so input strips overlap by kW - 1 columns.
    #   - vertical:   output advances by _H_TILE - kH + 1 rows,
    #     so input strips overlap by kH - 1 rows.
    H_step = _H_TILE - k_h + 1  # output rows advanced per vertical strip
    W_step = _W_TILE - k_w + 1  # output cols advanced per horizontal strip

    # Compute number of strips (at compile time from padded dimensions)
    # If the image fits in a single tile, num_strips = 1.
    num_hstrips = max(1, (Hp - k_h + H_step - 1) // H_step)
    num_wstrips = max(1, (Wp - k_w + W_step - 1) // W_step)

    # Effective tile size for this design (rounded up to handle padding)
    H_tile_eff = min(_H_TILE, Hp)
    W_tile_eff = min(_W_TILE, Wp)

    # --- Tensor types ---
    tensor_in_ty  = np.ndarray[(c * Hp * Wp,),    np.dtype[bfloat16]]
    tensor_w_ty   = np.ndarray[(c * k_h * k_w,),    np.dtype[bfloat16]]
    tensor_out_ty = np.ndarray[(c * Ho * Wo,),    np.dtype[bfloat16]]

    # Tile types: each acquire/release handles a *spatial strip* of one channel.
    # The kernel is called with C=1 always — channels are never mixed.
    #
    # TODO: when channels_per_column > 1, the worker loop serializes channels.
    # In the future, use double-buffered channel queues (ObjectFifo depth >
    # channels_per_column) to overlap DMA + compute across channels.
    tile_in_ty  = np.ndarray[(H_tile_eff * W_tile_eff,), np.dtype[bfloat16]]
    tile_w_ty   = np.ndarray[(k_h * k_w,),              np.dtype[bfloat16]]
    tile_out_ty = np.ndarray[(H_tile_eff * W_tile_eff,), np.dtype[bfloat16]]

    # ObjectFifos: one set per column
    of_ins = [ObjectFifo(tile_in_ty, name=f"in_{i}") for i in range(num_aie_columns)]
    of_ws  = [ObjectFifo(tile_w_ty, name=f"w_{i}") for i in range(num_aie_columns)]
    of_outs = [ObjectFifo(tile_out_ty, name=f"out_{i}") for i in range(num_aie_columns)]

    # Declare the C++ kernel — unchanged interface, called with C=1 per strip.
    conv_fn = Kernel(
        "depthwise_conv2d", "depthwise_conv.o",
        [tile_in_ty, tile_w_ty, tile_out_ty,
         np.int32, np.int32, np.int32, np.int32, np.int32],
    )

    # Worker task: each column processes its assigned channels.
    # Within each channel, the image is tiled spatially (H_TILE × W_TILE strips).
    def core_fn(col_idx):
        def body(of_in, of_w, of_out, kernel):
            offset_ch = col_idx  # first channel for this column
            for _ in range_(channels_per_column):
                # Load weights for this channel once per strip
                for hs in range_(num_hstrips):
                    H_act = min(_H_TILE, Hp - hs * H_step) if num_hstrips > 1 else Hp
                    if H_act < k_h:
                        continue  # strip too short, skip (should not happen)
                    for ws in range_(num_wstrips):
                        W_act = min(_W_TILE, Wp - ws * W_step) if num_wstrips > 1 else Wp
                        if W_act < k_w:
                            continue

                        elem_in = of_in.acquire(1)
                        elem_w = of_w.acquire(1)
                        elem_out = of_out.acquire(1)

                        kernel(elem_in, elem_w, elem_out,
                               1, H_act, W_act, k_h, k_w)

                        of_in.release(1)
                        of_w.release(1)
                        of_out.release(1)
                offset_ch += num_aie_columns
        return body

    my_workers = [
        Worker(
            core_fn(i),
            [of_ins[i].cons(), of_ws[i].cons(), of_outs[i].prod(), conv_fn],
        )
        for i in range(num_aie_columns)
    ]

    # TensorAccessPatterns for round-robin channel distribution + spatial strips.
    # Column i gets channels [i, i+num_columns, i+2*num_columns, ...].
    # Within each channel, the image is divided into H_TILE × W_TILE strips.
    in_taps = [
        [
            TensorAccessPattern(
                (1, c * Hp * Wp),
                offset=ch * Hp * Wp + hs * H_step * Wp + ws * W_step,
                sizes=[1, 1, 1,
                       min(_H_TILE, Hp - hs * H_step) if num_hstrips > 1 else Hp,
                       min(_W_TILE, Wp - ws * W_step) if num_wstrips > 1 else Wp],
                strides=[0, 0, 0, Wp, 1],
            )
            for ch in range(i, c, num_aie_columns)
            for hs in range(num_hstrips)
            for ws in range(num_wstrips)
            if (min(_H_TILE, Hp - hs * H_step) if num_hstrips > 1 else Hp) >= k_h
            and (min(_W_TILE, Wp - ws * W_step) if num_wstrips > 1 else Wp) >= k_w
        ]
        for i in range(num_aie_columns)
    ]
    w_taps = [
        TensorAccessPattern(
            (1, c * k_h * k_w),
            offset=i * k_h * k_w,
            sizes=[channels_per_column, 1, 1, k_h * k_w],
            strides=[num_aie_columns * k_h * k_w, 0, 0, 1],
        )
        for i in range(num_aie_columns)
    ]
    out_taps = [
        [
            TensorAccessPattern(
                (1, c * Ho * Wo),
                offset=ch * Ho * Wo + hs * H_step * Wo + ws * W_step,
                sizes=[1, 1, 1,
                       min(_H_TILE - k_h + 1, Ho - hs * H_step) if num_hstrips > 1 else Ho,
                       min(_W_TILE - k_w + 1, Wo - ws * W_step) if num_wstrips > 1 else Wo],
                strides=[0, 0, 0, Wo, 1],
            )
            for ch in range(i, c, num_aie_columns)
            for hs in range(num_hstrips)
            for ws in range(num_wstrips)
        ]
        for i in range(num_aie_columns)
    ]

    # Runtime: move data to/from the AIE array.
    # TAPs are lists of per-strip patterns, one per acquire.
    rt = Runtime()
    with rt.sequence(tensor_in_ty, tensor_w_ty, tensor_out_ty) as (A, B, C):
        rt.start(*my_workers)

        tg = rt.task_group()

        # Fill input strips and weights for each column
        for i in range(num_aie_columns):
            for tap in in_taps[i]:
                rt.fill(of_ins[i].prod(), A, tap=tap, task_group=tg)
            rt.fill(of_ws[i].prod(), B, tap=w_taps[i], task_group=tg)

        # Drain output strips from each column
        for i in range(num_aie_columns):
            for tap in out_taps[i]:
                rt.drain(of_outs[i].cons(), C, tap=tap, wait=True, task_group=tg)

        rt.finish_task_group(tg)

    # Place on device and generate MLIR
    return Program(dev, rt).resolve_program(SequentialPlacer())