# Depthwise Convolution on the AMD XDNA2 NPU

**A bf16 3×3 depthwise conv2d operator for AMD Ryzen AI NPUs, built with
IRON — from a scalar single-core kernel up to a vectorized design running
on all 8 columns (32 AIE cores).**

This folder holds the documentation of **my project** (branch
`socdaml-giulio`). The rest of the repository is the upstream
[IRON](https://github.com/amd/IRON) framework; its own docs (top-level
`README.md`, `AGENTS.md`, `CONTRIBUTING.md`, …) are not about this work.

## Where to start

| Read this | If you want |
|---|---|
| [FINAL_BRIEF.md](FINAL_BRIEF.md) | The results in one place: scalar vs vectorized vs multi-column, and NPU vs CPU (1 core and all 24 threads). |
| [DEPTHWISE_CONV_STATUS.md](DEPTHWISE_CONV_STATUS.md) | How the design works: hardware model, each operator's architecture, tiling, hardware limits. |
| [PERFORMANCE.md](PERFORMANCE.md) | Full benchmark analysis and methodology. |
| [FILES_MODIFIED.md](FILES_MODIFIED.md) | Every file I added or changed in the repo, and what it contains. |

## Final operator

`iron/operators/vector_conv2d_dw_multicolumn/` — channels are split across
up to 8 NPU columns; inside each column, a MemTile fans the (host-padded,
HWC) image out to 4 cores by row strips, with W-tiling for wide images.
Kernel: `aie_kernels/aie2p/vector_conv2d_dw_mc.cc` (32 channels per vector
MAC).

## Running it

```bash
source /opt/xilinx/xrt/setup.sh
source ironenv/bin/activate          # or the project's micromamba env
pytest iron/operators/vector_conv2d_dw_multicolumn/ -m "not extensive"
python benchmarks/bench_full_sweep.py              # NPU benchmark sweep
cmake -S benchmarks/cpu_libtorch -B benchmarks/cpu_libtorch/build \
      -DCMAKE_PREFIX_PATH=<libtorch> && cmake --build benchmarks/cpu_libtorch/build
```
