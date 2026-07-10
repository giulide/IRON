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
from iron.operators.vector_conv2d_dw_multicore.tiling import compute_tiling


@dataclass
class VectorConv2DDWMC(MLIROperator):
    """AIE-accelerated depthwise 2D convolution, vectorized across channels
    (HWC layout), parallelized across the 4 cores of a single column.

    Same math and channel-vectorization as the single-column VectorConv2DDW
    (each channel has its own kH*kW filter; VEC=32 channels per aie::mac; the
    channel count is rounded up to a multiple of VEC — see `channels_padded`).
    The image is split into contiguous bands of output rows, one per core; each
    core streams its band through the column MemTile (L2) as L1-sized row strips
    with receptive-field overlap, so arbitrarily tall images are handled without
    exceeding the L1 budget. Padding is applied in-kernel using global
    coordinates, so it lands only on the true image borders, not band/strip
    boundaries.
    """

    h: int  # input height
    w: int  # input width
    c: int  # number of channels
    k_h: int  # kernel height
    k_w: int  # kernel width
    padding: int = 0
    num_cores: int = 4  # cores in one column of the AIE array
    context: AIEContext | None = field(default=None, repr=False)

    # Native AIE2P bf16 vector width (512-bit / 16-bit) — must match VEC in
    # aie_kernels/aie2p/vector_conv2d_dw_mc.cc and _VEC in design.py.
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
    }

    def __post_init__(self):
        super().__init__(context=self.context)

    @property
    def channels_padded(self) -> int:
        """Channel count rounded up to a multiple of VEC (see class docstring)."""
        return (self.c + self._VEC - 1) // self._VEC * self._VEC

    @property
    def tiling(self) -> dict:
        """Padded tiling geometry shared with design.py (see tiling.py)."""
        return compute_tiling(
            self.h, self.w, self.c, self.k_h, self.k_w, self.padding, self.num_cores
        )

    def get_arg_spec(self) -> list[AIERuntimeArgSpec]:
        # Buffers are in *padded* space: the host pre-pads the input (H and W)
        # and pads the output rows up to a whole number of macro-tiles. The
        # kernel then needs no boundary handling and every DMA is regular.
        g = self.tiling
        cp = g["c_padded"]
        return [
            AIERuntimeArgSpec("in", (g["hp_padded"] * g["wp"] * cp,)),  # padded input
            AIERuntimeArgSpec("in", (self.k_h * self.k_w * cp,)),       # weights
            AIERuntimeArgSpec("out", (g["H_out_pad"] * g["W_out"] * cp,)),  # padded output
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
        ]

    def get_mlir_artifact(self) -> PythonGeneratedMLIRArtifact:
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "conv2d_dw_multicore",
                tuple(self._mlir_callback_args()),
            ),
        )

    def get_kernel_artifacts(self) -> list[KernelObjectArtifact]:
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
