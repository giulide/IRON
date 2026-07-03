# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ml_dtypes import bfloat16
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_


def conv2d_dw_design(dev, c, h, w, k_h, k_w, padding, num_aie_columns):
    H_out = h + 2 * padding - k_h + 1
    W_out = w + 2 * padding - k_w + 1

    # Number of channels per column (round-robin distribution)
    channels_per_column = (c + num_aie_columns - 1) // num_aie_columns

    # Tensor types (flat 1D, CHW layout)
    tensor_in_ty = np.ndarray[(c * h * w,), np.dtype[bfloat16]]
    tensor_w_ty = np.ndarray[(c * k_h * k_w,), np.dtype[bfloat16]]
    tensor_out_ty = np.ndarray[(c * H_out * W_out,), np.dtype[bfloat16]]

    # Tile types: one column processes channels_per_column channels
    tile_in_ty = np.ndarray[(channels_per_column * h * w,), np.dtype[bfloat16]]
    tile_w_ty = np.ndarray[(channels_per_column * k_h * k_w,), np.dtype[bfloat16]]
    tile_out_ty = np.ndarray[(channels_per_column * H_out * W_out,), np.dtype[bfloat16]]

    # ObjectFifos: one set per column
    of_ins = [ObjectFifo(tile_in_ty, name=f"in_{i}") for i in range(num_aie_columns)]
    of_ws  = [ObjectFifo(tile_w_ty, name=f"w_{i}") for i in range(num_aie_columns)]
    of_outs = [ObjectFifo(tile_out_ty, name=f"out_{i}") for i in range(num_aie_columns)]

    # Declare the C++ kernel (each core runs the same kernel on its slice)
    conv_fn = Kernel(
        "depthwise_conv2d", "depthwise_conv.o",
        [tile_in_ty, tile_w_ty, tile_out_ty,
         np.int32, np.int32, np.int32, np.int32, np.int32, np.int32],
    )

    # Worker task: each column processes its assigned channels
    def core_body(of_in, of_w, of_out, kernel):
        elem_in = of_in.acquire(1)
        elem_w = of_w.acquire(1)
        elem_out = of_out.acquire(1)
        kernel(elem_in, elem_w, elem_out, channels_per_column, h, w, k_h, k_w, padding)
        of_in.release(1)
        of_w.release(1)
        of_out.release(1)

    my_workers = [
        Worker(
            core_body,
            [of_ins[i].cons(), of_ws[i].cons(), of_outs[i].prod(), conv_fn],
        )
        for i in range(num_aie_columns)
    ]

    # TensorAccessPatterns for round-robin channel distribution.
    # Column i gets channels [i, i+num_columns, i+2*num_columns, ...].
    # Each channel occupies flat elements: H*W for input, kH*kW for weights, H_out*W_out for output.
    in_taps = [
        TensorAccessPattern(
            (1, c * h * w),
            offset=i * h * w,
            sizes=[channels_per_column, 1, 1, h * w],
            strides=[num_aie_columns * h * w, 0, 0, 1],
        )
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
        TensorAccessPattern(
            (1, c * H_out * W_out),
            offset=i * H_out * W_out,
            sizes=[channels_per_column, 1, 1, H_out * W_out],
            strides=[num_aie_columns * H_out * W_out, 0, 0, 1],
        )
        for i in range(num_aie_columns)
    ]

    # Runtime: move data to/from the AIE array
    rt = Runtime()
    with rt.sequence(tensor_in_ty, tensor_w_ty, tensor_out_ty) as (A, B, C):
        rt.start(*my_workers)

        tg = rt.task_group()

        # Fill input and weights for each column
        for i in range(num_aie_columns):
            rt.fill(of_ins[i].prod(), A, tap=in_taps[i], task_group=tg)
            rt.fill(of_ws[i].prod(), B, tap=w_taps[i], task_group=tg)

        # Drain output from each column
        for i in range(num_aie_columns):
            rt.drain(of_outs[i].cons(), C, tap=out_taps[i], wait=True, task_group=tg)

        rt.finish_task_group(tg)

    # Place on device and generate MLIR
    return Program(dev, rt).resolve_program(SequentialPlacer())