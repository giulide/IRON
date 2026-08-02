# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Depthwise conv parallelized across up to 8 NPU columns by CHANNEL slicing.

Depthwise conv has no cross-channel dependency, so column `col` independently
computes the full h x w spatial extent for its own disjoint slice of `c`
channels (the SAME `c` for every column -- total logical channels handled
= c * num_columns). This reuses the tested single-column, 4-core design
(`vector_conv2d_dw_multicore._build_column`) verbatim as the per-column unit:
no new spatial-tiling logic, no cross-column data sharing, every per-column
hardware guard (MAX_ROW_IN, MAX_W_SLICES -- see that module's docstring)
still applies independently per column exactly as it does for one column.

The 3 top-level tensors are `num_columns` independently-padded HWC blocks
stacked back-to-back (NOT a literal interleaved [h, w, c*num_columns]
tensor) -- see `_build_column`'s docstring for why, and its `num_columns`/
`*_base_offset` params which this module exists to drive.
"""

from ml_dtypes import bfloat16
import numpy as np

from aie.iron import Program, Runtime
from aie.iron.placers import SequentialPlacer

from iron.operators.vector_conv2d_dw_multicore.design import (
    _build_column,
    _check_hw_limits,
    _make_conv_kernel,
)
from iron.operators.vector_conv2d_dw_multicore.tiling import compute_tiling


def conv2d_dw_multicolumn(dev, h, w, c, k_h, k_w, padding, num_cores, num_columns):
    # Every column shares identical (h, w, c, k_h, k_w, padding, num_cores),
    # so one compute_tiling() call is representative of every column: safe
    # to hoist out of the loop (pure arithmetic, no I/O/randomness).
    g = compute_tiling(h, w, c, k_h, k_w, padding, num_cores)
    for col in range(num_columns):
        _check_hw_limits(g, col, w, c)

    c_padded = g["c_padded"]
    per_col_in_elems = g["hp_padded"] * g["wp_padded"] * c_padded
    per_col_w_elems = k_h * k_w * c_padded
    per_col_out_elems = g["H_out_pad"] * g["W_out_pad"] * c_padded

    tensor_in_ty = np.ndarray[(num_columns * per_col_in_elems,), np.dtype[bfloat16]]
    tensor_w_ty = np.ndarray[(num_columns * per_col_w_elems,), np.dtype[bfloat16]]
    tensor_out_ty = np.ndarray[(num_columns * per_col_out_elems,), np.dtype[bfloat16]]

    # ONE shared Kernel object for every column's Workers -- see
    # _build_column's docstring: a fresh Kernel() per column, even with an
    # identical name/signature, makes MLIR verification fail with
    # "redefinition of symbol".
    conv_fn = _make_conv_kernel(g, k_h, k_w)

    rt = Runtime()
    with rt.sequence(tensor_in_ty, tensor_w_ty, tensor_out_ty) as (A, B, Y):
        tg = rt.task_group()
        for col in range(num_columns):
            _build_column(
                rt, tg, col, h, w, c, k_h, k_w, padding, num_cores, conv_fn, A, B, Y,
                num_columns=num_columns,
                in_base_offset=col * per_col_in_elems,
                w_base_offset=col * per_col_w_elems,
                out_base_offset=col * per_col_out_elems,
            )
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())
