# Files Modified in This Project

Every file added or changed by me (`giulide`) on branch `socdaml-giulio`,
relative to the upstream base branch `devel`. Built from the git history
(15 commits, `8ceb5f3` → `04d3ede`, plus the final documentation commit).
Everything else in the repository is upstream IRON code, untouched.

Regenerate the raw list with:

```bash
git diff --stat devel..socdaml-giulio
git log --author=giulide --stat devel..socdaml-giulio
```

## Implementation progression

```
scalar_conv2d → scalar_conv2d_dw → vector_conv2d_dw → vector_conv2d_dw_multicore → vector_conv2d_dw_multicolumn
 (1 channel)     (C channels,       (SIMD across       (1 column,                   (up to 8 columns,
                  scalar, 1 core)    channels, 1 core)  4 cores)                     32 cores) ← final
```

## 1. Operators — `iron/operators/`

Each operator follows the IRON layout: `op.py` (Python interface, an
`MLIROperator` subclass), `design.py` (NPU dataflow: ObjectFifos, Workers,
runtime sequence), `reference.py` (CPU golden model), `test.py` (pytest:
build, run on NPU, compare against reference).

### `scalar_conv2d/` — first step: plain 2D convolution
| File | Content |
|---|---|
| `op.py` | `ScalarConv2D` operator: single-channel conv2d (H, W, kernel size, padding). |
| `design.py` | One-core, untiled design; pads the weight buffer to a multiple of 4 elements (Shim DMA requirement). |
| `reference.py` | PyTorch `conv2d` reference for one channel. |
| `test.py` | Correctness tests over a few image/kernel sizes. |

### `scalar_conv2d_dw/` — scalar depthwise baseline
| File | Content |
|---|---|
| `op.py` | `ScalarConv2DDW` operator: depthwise conv (groups = C), no vectorization. |
| `design.py` | Single core, with H-only spatial tiling (row strips that fit L1) so it can reach the same image sizes as the multi-column design (up to 366×366). Host-side zero padding. |
| `tiling.py` | Pure-Python geometry for the H-tiling (strip heights, overlaps, L1 budget). |
| `reference.py` | PyTorch depthwise reference. |
| `test.py` | Correctness tests, including tiled sizes. Used as the speedup baseline. |

### `vector_conv2d_dw/` — vectorized, single core
| File | Content |
|---|---|
| `op.py` | `VectorConv2DDW` operator; rounds channels up to a multiple of 32 (vector width). |
| `design.py` | Single-core, untiled design (whole image in one DMA → only works while it fits in 64 KB L1). |
| `reference.py` | Depthwise reference in HWC layout. |
| `test.py` | Correctness tests over channel counts / small image sizes. |

### `vector_conv2d_dw_multicore/` — one column, 4 cores
| File | Content |
|---|---|
| `tiling.py` | `compute_tiling`: padded shapes, W-tiling (`num_w_sub`), per-core row distribution. Holds the hardware-limit constants `MAX_ROW_IN`, `MAX_W_SLICES`. No AIE imports. |
| `design.py` | Builds one column: MemTile + 4 Workers wired with `split()`/`join()`, all tiles explicitly placed; Shim channels kept within the native 2-in/2-out budget. `_check_hw_limits` raises clear errors; `_build_column` is reused by the multi-column design. |
| `op.py` | `VectorConv2DDWMC` operator wrapper. |
| `reference.py` | Depthwise reference (HWC). |
| `test.py` | Correctness tests including W-tiled and odd sizes. |

### `vector_conv2d_dw_multicolumn/` — final design, up to 8 columns
| File | Content |
|---|---|
| `design.py` | Instantiates the single-column building block on up to 8 columns, slicing **channels** across columns. |
| `op.py` | `VectorConv2DDWMColumn` operator: chooses column count / channels per column, validates hardware limits. |
| `test.py` | Correctness tests across channel counts (32–1024) and image sizes; helpers (`pad_channels`, `build_padded_input`) reused by the benchmarks. |

### `__init__.py`
Registers `ScalarConv2D`, `ScalarConv2DDW`, `VectorConv2DDW`.

### Removed during the project
`conv2d/` (renamed into `scalar_conv2d/`) and `conv2d_dw/` (first, never
completed depthwise attempt; deleted in `7205466`, superseded by the
multicore/multicolumn operators).

## 2. AIE kernels — `aie_kernels/`

| File | Content | Used by |
|---|---|---|
| `generic/scalar_conv.cc` | Scalar single-channel conv2d with in-kernel zero padding. | `scalar_conv2d` |
| `generic/scalar_conv2d_dw.cc` | Scalar depthwise conv over an already-padded tile (padding=0, no bounds checks). | `scalar_conv2d_dw` |
| `aie2p/vector_conv2d_dw.cc` | Vectorized depthwise, HWC layout: 32 channels per `aie::mac`; takes global-coordinate offsets so padding is applied only at true image borders. | `vector_conv2d_dw` |
| `aie2p/vector_conv2d_dw_mc.cc` | Multi-core variant: host-padded input, one call computes a strip of output rows; no bounds checks. | `vector_conv2d_dw_multicore`, `vector_conv2d_dw_multicolumn` |
| `generic/depthwise_conv.cc` | Early CHW depthwise kernel (scalar). | Legacy — was used only by the removed `conv2d_dw` |
| `aie2p/depthwise_conv.cc` | Early vectorized CHW depthwise kernel. | Legacy — was used only by the removed `conv2d_dw` |

## 3. Benchmarks — `benchmarks/`

| File | Content |
|---|---|
| `bench_full_sweep.py` | NPU sweep: scalar vs vector vs multi-column over channels 32–1024 and growing image sizes; source of the NPU tables. |
| `bench_cpu_vs_npu_grid.py` | Early CPU (PyTorch, Python, single thread) vs NPU comparison on the same grid. Superseded by the C++ LibTorch bench. |
| `cpu_libtorch/bench_libtorch.cpp` | CPU baseline in C++ against LibTorch (no Python in the timed path), bf16, channels_last. Final version uses **all CPU threads** (`hardware_concurrency`) with a warm-up phase; the single-thread numbers come from the previous version (`at::set_num_threads(1)`, commit `04d3ede`). |
| `cpu_libtorch/CMakeLists.txt` | Build file for the LibTorch benchmark. |

Plot images in `benchmarks/figures/` are local only and not committed.

## 4. Documentation — `npu_depthwise_conv_project/`

| File | Content |
|---|---|
| `README.md` | Entry point of this folder. |
| `FINAL_BRIEF.md` | Final results: execution times for scalar, vectorized, multi-column; NPU vs 1 CPU core; NPU vs all 24 CPU threads. |
| `PERFORMANCE.md` | Detailed benchmark analysis (methodology, speedups, max processable sizes, limits). |
| `DEPTHWISE_CONV_STATUS.md` | Technical design document: hardware, architecture of each operator, tiling, hardware limits, known issues. |
| `FILES_MODIFIED.md` | This file. |
