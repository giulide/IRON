# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ml_dtypes import bfloat16
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_

# Native AIE2P bf16 vector width (512-bit / 16-bit) — must match VEC in
# aie_kernels/aie2p/vector_conv2d_dw.cc.
_VEC = 32


def conv2d_dw_vector(dev, h, w, c, k_h, k_w, padding):
    H_out = h + 2 * padding - k_h + 1
    W_out = w + 2 * padding - k_w + 1

    # The kernel's vectorized loop has no scalar tail: round the channel
    # count up to a multiple of VEC so every iteration is a full vector op.
    # The extra channels are zero-weighted (see op.py/test.py padding) so
    # they don't affect the real output channels.
    c_padded = (c + _VEC - 1) // _VEC * _VEC

    # Tensor types (HWC layout, channel-innermost)
    tensor_in_ty = np.ndarray[(h * w * c_padded,), np.dtype[bfloat16]]
    tensor_w_ty = np.ndarray[(k_h * k_w * c_padded,), np.dtype[bfloat16]]
    tensor_out_ty = np.ndarray[(H_out * W_out * c_padded,), np.dtype[bfloat16]]

    # Use the same types as tile types (no tiling for this 1-column design)
    tile_in_ty = tensor_in_ty
    tile_w_ty = tensor_w_ty
    tile_out_ty = tensor_out_ty

    # ObjectFifos: one for input, one for weights, one for output
    of_in = ObjectFifo(tile_in_ty, name="in")
    of_w = ObjectFifo(tile_w_ty, name="weights")
    of_out = ObjectFifo(tile_out_ty, name="out")

    # Declare the C++ kernel
    conv_fn = Kernel(
        "vector_conv2d_dw",
        "vector_conv2d_dw.o",
        [
            tile_in_ty,
            tile_w_ty,
            tile_out_ty,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )

    # Worker task: runs once on a compute tile
    def core_body(of_in, of_w, of_out, kernel):
        elem_in = of_in.acquire(1)
        elem_w = of_w.acquire(1)
        elem_out = of_out.acquire(1)
        kernel(elem_in, elem_w, elem_out, h, w, c_padded, k_h, k_w, padding)
        of_in.release(1)
        of_w.release(1)
        of_out.release(1)

    worker = Worker(core_body, [of_in.cons(), of_w.cons(), of_out.prod(), conv_fn])

    # Runtime: move data to/from the AIE array
    rt = Runtime()
    with rt.sequence(tensor_in_ty, tensor_w_ty, tensor_out_ty) as (A, B, Y):
        rt.start(worker)
        rt.fill(of_in.prod(), A)
        rt.fill(of_w.prod(), B)
        rt.drain(of_out.cons(), Y, wait=True)

    # Place on device and generate MLIR
    return Program(dev, rt).resolve_program(SequentialPlacer())
