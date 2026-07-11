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
class VectorConv2DDW(MLIROperator):
    """AIE-accelerated depthwise 2D convolution, vectorized across channels
    (HWC layout, single column).

    Each channel has its own kH*kW filter (no channel mixing). Vectorization
    is across the C dimension (VEC=32 channels processed per aie::mac), which
    makes this design effective even for small H/W, unlike a spatially
    (W-)vectorized design that needs W_out >= VEC.

    The kernel has no scalar tail path: the channel count is rounded up to a
    multiple of VEC (see `channels_padded`), and callers must zero-pad the
    weights for the extra channels so they don't affect real outputs.
    """

    h: int  # input height
    w: int  # input width
    c: int  # number of channels
    k_h: int  # kernel height
    k_w: int  # kernel width
    padding: int = 0
    context: AIEContext | None = field(default=None, repr=False)

    # Native AIE2P bf16 vector width (512-bit / 16-bit) — must match VEC in
    # aie_kernels/aie2p/vector_conv2d_dw.cc and _VEC in design.py.
    _VEC: ClassVar[int] = 32

    # Override default name aliases: use shorter aliases for conv2d fields
    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "h": "H",
        "w": "W",
        "c": "C",
        "k_h": "kh",
        "k_w": "kw",
        "padding": "pad",
    }

    def __post_init__(self):
        super().__init__(context=self.context)

    @property
    def channels_padded(self) -> int:
        """Channel count rounded up to a multiple of VEC (see class docstring)."""
        return (self.c + self._VEC - 1) // self._VEC * self._VEC

    def get_arg_spec(self) -> list[AIERuntimeArgSpec]:
        H_out = self.h + 2 * self.padding - self.k_h + 1
        W_out = self.w + 2 * self.padding - self.k_w + 1
        cp = self.channels_padded
        return [
            AIERuntimeArgSpec("in", (self.h * self.w * cp,)),  # input (HWC, channel-padded)
            AIERuntimeArgSpec("in", (self.k_h * self.k_w * cp,)),  # weights (HWC, channel-padded)
            AIERuntimeArgSpec("out", (H_out * W_out * cp,)),  # output (HWC, channel-padded)
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
        ]

    def get_mlir_artifact(self) -> PythonGeneratedMLIRArtifact:
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "conv2d_dw_vector",
                tuple(self._mlir_callback_args()),
            ),
        )

    def get_kernel_artifacts(self) -> list[KernelObjectArtifact]:
        return [
            KernelObjectArtifact(
                "vector_conv2d_dw.o",
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir
                        / "aie_kernels"
                        / "aie2p"
                        / "vector_conv2d_dw.cc"
                    )
                ],
            )
        ]
