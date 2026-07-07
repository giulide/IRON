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
class Conv2DDW(MLIROperator):
    """AIE-accelerated depthwise 2D convolution (multi-channel, multi-column)"""

    c: int  # number of channels
    h: int  # input height
    w: int  # input width
    k_h: int  # kernel height
    k_w: int  # kernel width
    padding: int = 0
    num_aie_columns: int = field(default=8)
    context: AIEContext | None = field(default=None, repr=False)

    # Override default name aliases
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
        if self.num_aie_columns > aie_utils.get_current_device().cols:
            raise ValueError(
                f"num_aie_columns ({self.num_aie_columns}) exceeds device columns"
            )
        super().__init__(context=self.context)

    def get_arg_spec(self) -> list[AIERuntimeArgSpec]:
        # Input is externally padded: C * Hp * Wp
        Hp = self.h + 2 * self.padding
        Wp = self.w + 2 * self.padding
        Ho = Hp - self.k_h + 1
        Wo = Wp - self.k_w + 1
        return [
            AIERuntimeArgSpec("in", (self.c * Hp * Wp,)),          # padded input
            AIERuntimeArgSpec("in", (self.c * self.k_h * self.k_w,)),  # weights
            AIERuntimeArgSpec("out", (self.c * Ho * Wo,)),         # output
        ]

    def _mlir_callback_args(self) -> list[Any]:
        """Arguments forwarded to the design.py callback."""
        return [aie_utils.get_current_device(), self.c, self.h, self.w,
                self.k_h, self.k_w, self.padding, self.num_aie_columns]

    def get_mlir_artifact(self) -> PythonGeneratedMLIRArtifact:
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "conv2d_dw_design",
                tuple(self._mlir_callback_args()),
            ),
        )

    def get_kernel_artifacts(self) -> list[KernelObjectArtifact]:
        return [
            KernelObjectArtifact(
                "depthwise_conv.o",
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir
                        / "aie_kernels"
                        / "aie2p"
                        / "depthwise_conv.cc"
                    )
                ],
            )
        ]
