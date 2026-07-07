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


@dataclass
class ScalarConv2D(MLIROperator):
    """AIE-accelerated scalar 2D convolution (single channel, single column)"""

    h: int  # input height
    w: int  # input width
    k_h: int  # kernel height
    k_w: int  # kernel width
    padding: int = 0
    context: AIEContext | None = field(default=None, repr=False)

    # Override default name aliases: use shorter aliases for conv2d fields
    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
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
        """Weight buffer size, rounded up to a multiple of 4 elements.

        The shim DMA requires transfer lengths to be 4-byte-aligned (2 bf16
        elements); we round up to 4 elements to match the padding done in
        design.py. Callers must zero-pad the raw k_h*k_w weights to this size.
        """
        k_size = self.k_h * self.k_w
        return (k_size + 3) // 4 * 4

    def get_arg_spec(self) -> list[AIERuntimeArgSpec]:
        H_out = self.h + 2 * self.padding - self.k_h + 1
        W_out = self.w + 2 * self.padding - self.k_w + 1
        return [
            AIERuntimeArgSpec("in", (self.h * self.w,)),  # input
            AIERuntimeArgSpec("in", (self.weight_size,)),  # weights (padded)
            AIERuntimeArgSpec("out", (H_out * W_out,)),  # output
        ]

    def _mlir_callback_args(self) -> list[Any]:
        """Arguments forwarded to the design.py callback."""
        return [
            aie_utils.get_current_device(),
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
                "conv2d_scalar",
                tuple(self._mlir_callback_args()),
            ),
        )

    def get_kernel_artifacts(self) -> list[KernelObjectArtifact]:
        return [
            KernelObjectArtifact(
                "scalar_conv.o",
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir
                        / "aie_kernels"
                        / "generic"
                        / "scalar_conv.cc"
                    )
                ],
            )
        ]
