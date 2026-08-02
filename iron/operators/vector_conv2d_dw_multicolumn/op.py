# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, ClassVar

import aie.utils as aie_utils

from iron.common import (
    MLIROperator,
    AIERuntimeArgSpec,
    KernelObjectArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
    DesignGenerator,
)
from iron.common.context import AIEContext
from iron.operators.vector_conv2d_dw_multicore.design import _check_hw_limits
from iron.operators.vector_conv2d_dw_multicore.tiling import compute_tiling


@dataclass
class VectorConv2DDWMColumn(MLIROperator):
    """AIE-accelerated depthwise 2D convolution, parallelized across up to 8
    NPU columns by CHANNEL slicing (HWC layout).

    Depthwise conv has no cross-channel dependency, so column `col` handles
    the full h x w spatial extent for its own disjoint slice of `c` channels
    -- the SAME `c` (and h, w, k_h, k_w, padding, num_cores) for every
    column, so total logical channels handled = `c * num_columns`. Each
    column reuses, verbatim, the tested single-column 4-core design
    (`VectorConv2DDWMC` / `vector_conv2d_dw_multicore._build_column`) as an
    independent unit -- see that module's docstring for the round-robin H
    trick, W-tiling, and the MAX_ROW_IN/MAX_W_SLICES hardware limits, which
    apply per column exactly as they do for a single column.
    """

    h: int  # input height
    w: int  # input width
    c: int  # channels PER COLUMN (total channels = c * num_columns)
    k_h: int  # kernel height
    k_w: int  # kernel width
    padding: int = 0
    num_cores: int = 4  # cores in one column of the AIE array
    num_columns: int = 1
    context: AIEContext | None = field(default=None, repr=False)

    # Native AIE2P bf16 vector width (512-bit / 16-bit) -- must match VEC in
    # aie_kernels/aie2p/vector_conv2d_dw_mc.cc and _VEC in tiling.py.
    _VEC: ClassVar[int] = 32

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "h": "H",
        "w": "W",
        "c": "C",
        "k_h": "kh",
        "k_w": "kw",
        "padding": "pad",
        "num_cores": "cores",
        "num_columns": "col",
    }

    def __post_init__(self):
        dev = aie_utils.get_current_device()
        if self.num_columns < 1 or self.num_columns > dev.cols:
            raise ValueError(
                f"VectorConv2DDWMColumn: num_columns ({self.num_columns}) must be "
                f"between 1 and this device's column count ({dev.cols})."
            )
        # Explicit, early hardware-limit check (rather than only inheriting it
        # implicitly from the reused per-column tiling code deep inside MLIR
        # generation): every column shares the same (h, w, c, k_h, k_w,
        # padding, num_cores), so one compute_tiling() call, checked once per
        # column for a clear per-column error message, is sufficient.
        g = self.tiling
        for col in range(self.num_columns):
            _check_hw_limits(g, col, self.w, self.c)
        super().__init__(context=self.context)

    @property
    def channels_padded(self) -> int:
        """Per-column channel count rounded up to a multiple of VEC (see class docstring)."""
        return (self.c + self._VEC - 1) // self._VEC * self._VEC

    @property
    def tiling(self) -> dict:
        """Per-column padded tiling geometry, shared with design.py (see
        vector_conv2d_dw_multicore/tiling.py) -- identical for every column."""
        return compute_tiling(
            self.h, self.w, self.c, self.k_h, self.k_w, self.padding, self.num_cores
        )

    def get_arg_spec(self) -> list[AIERuntimeArgSpec]:
        # Buffers are `num_columns` independently-padded per-column HWC
        # blocks stacked back-to-back (see design.py module docstring), NOT
        # a literal interleaved [h, w, c*num_columns] tensor.
        g = self.tiling
        cp = g["c_padded"]
        n = self.num_columns
        return [
            AIERuntimeArgSpec("in", (n * g["hp_padded"] * g["wp_padded"] * cp,)),
            AIERuntimeArgSpec("in", (n * self.k_h * self.k_w * cp,)),
            AIERuntimeArgSpec("out", (n * g["H_out_pad"] * g["W_out_pad"] * cp,)),
        ]

    def _mlir_callback_args(self) -> list[Any]:
        """Arguments forwarded to the design.py callback."""
        return [
            aie_utils.get_current_device(),
            self.h,
            self.w,
            self.c,
            self.k_h,
            self.k_w,
            self.padding,
            self.num_cores,
            self.num_columns,
        ]

    def get_mlir_artifact(self) -> PythonGeneratedMLIRArtifact:
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "conv2d_dw_multicolumn",
                tuple(self._mlir_callback_args()),
            ),
        )

    def get_kernel_artifacts(self) -> list[KernelObjectArtifact]:
        # Same per-core kernel binary as the single-column operator -- the
        # compiled AIE kernel is identical regardless of which column runs
        # it, so it's built from the SAME source, not duplicated/renamed.
        return [
            KernelObjectArtifact(
                "vector_conv2d_dw_mc.o",
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir
                        / "aie_kernels"
                        / "aie2p"
                        / "vector_conv2d_dw_mc.cc"
                    )
                ],
            )
        ]
