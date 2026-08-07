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
from iron.operators.scalar_conv2d_dw.tiling import compute_tiling


@dataclass
class ScalarConv2DDW(MLIROperator):
    """AIE-accelerated scalar depthwise 2D convolution (CHW layout, single core).

    Each channel has its own kH*kW filter (no channel mixing). H-tiled (see
    design.py/tiling.py module docstrings) to handle images past the single
    shot L1 ceiling -- intended for use at c=1, where W-tiling is never
    needed (see tiling.py).
    """

    c: int  # number of channels
    h: int  # input height
    w: int  # input width
    k_h: int  # kernel height
    k_w: int  # kernel width
    padding: int = 0
    context: AIEContext | None = field(default=None, repr=False)

    # Override default name aliases: use shorter aliases for conv2d fields
    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "c": "C",
        "h": "H",
        "w": "W",
        "k_h": "kh",
        "k_w": "kw",
        "padding": "pad",
    }

    def __post_init__(self):
        super().__init__(context=self.context)

    @property
    def weight_size(self) -> int:
        """Weight buffer size (c * k_h * k_w), rounded up to a multiple of 4 elements.

        The shim DMA requires transfer lengths to be 4-byte-aligned (2 bf16
        elements); we round up to 4 elements to match the padding done in
        design.py. Callers must zero-pad the raw c*k_h*k_w weights to this size.
        """
        k_size = self.c * self.k_h * self.k_w
        return (k_size + 3) // 4 * 4

    @property
    def tiling(self) -> dict:
        """Padded H-tiling geometry shared with design.py (see tiling.py)."""
        return compute_tiling(self.h, self.w, self.c, self.k_h, self.k_w, self.padding)

    def get_arg_spec(self) -> list[AIERuntimeArgSpec]:
        # Buffers are in *padded* space: the host pre-pads the input (H
        # only -- see tiling.py) and pads the output rows up to whole
        # sub-tiles. The kernel then needs no boundary handling.
        g = self.tiling
        return [
            AIERuntimeArgSpec("in", (self.c * g["hp_padded"] * g["wp"],)),  # padded input (CHW)
            AIERuntimeArgSpec("in", (self.weight_size,)),  # weights (padded, per-channel)
            AIERuntimeArgSpec("out", (self.c * g["H_out_pad"] * g["W_out"],)),  # padded output (CHW)
        ]

    def _mlir_callback_args(self) -> list[Any]:
        """Arguments forwarded to the design.py callback."""
        return [
            aie_utils.get_current_device(),
            self.c,
            self.h,
            self.w,
            self.k_h,
            self.k_w,
            self.padding,
        ]

    def get_mlir_artifact(self) -> PythonGeneratedMLIRArtifact:
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "conv2d_scalar_dw",
                tuple(self._mlir_callback_args()),
            ),
        )

    def get_kernel_artifacts(self) -> list[KernelObjectArtifact]:
        return [
            KernelObjectArtifact(
                "scalar_conv2d_dw.o",
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir
                        / "aie_kernels"
                        / "generic"
                        / "scalar_conv2d_dw.cc"
                    )
                ],
            )
        ]
