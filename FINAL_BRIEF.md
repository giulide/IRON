# Final Brief — Depthwise Convolution on AMD XDNA2 NPU: Performance Summary

This document collects the execution times of the three implementations
built for this project — **scalar** (no parallelism), **vectorized**
(single core, SIMD), and **multi-column** (parallel across up to 8 NPU
columns) — on the same test cases, to show the improvement at each step.

**Test cases**: channel counts from 32 up to 1024, and image sizes growing
from 16×16 up to the largest size each configuration can process (366×366
for the smaller channel counts, shrinking as channel count grows, down to
78×78 at 1024 channels). Kernel 3×3, padding 1. Same grid used throughout
`PERFORMANCE.md`.

This version covers all three NPU implementations plus a comparison
against a single CPU core.

## 1. Scalar kernel (baseline, no parallelism)

**Method**: the scalar kernel handles every channel independently, so we
measured the real NPU execution time for **one channel** at each image
size (10 warm-up runs, then 50 timed runs averaged), and multiplied by the
channel count to get the total. The per-channel number is a real hardware
measurement; only the multiplication is an estimate — and a conservative
one, since it doesn't add the extra time real back-to-back launches would
cost, so actual multi-channel time would likely be slightly higher.

**Execution times (ms)**

| Channels | 16×16 | 64×64 | 78×78 | 114×114 | 174×174 | 366×366 |
|---:|---:|---:|---:|---:|---:|---:|
| 32   | 4.66   | 54.94    | 70.44    | 91.05    | 178.72   | 750.23    |
| 64   | 9.32   | 109.89   | 140.87   | 182.11   | 357.43   | 1,500.47  |
| 96   | 13.97  | 164.83   | 211.31   | 273.16   | 536.15   | 2,250.70  |
| 128  | 18.63  | 219.78   | 281.75   | 364.21   | 714.86   | 3,000.93  |
| 256  | 37.26  | 439.56   | 563.50   | 728.42   | 1,429.72 | 6,001.87  |
| 512  | 74.52  | 879.11   | 1,126.99 | 1,456.84 | 2,859.45 | 12,003.74 |
| 768  | 111.79 | 1,318.67 | 1,690.49 | 2,185.27 | 4,289.17 | 18,005.61 |
| 1024 | 149.05 | 1,758.23 | 2,253.99 | 2,913.69 | 5,718.89 | 24,007.48 |

## 2. Vectorized kernel (single core, SIMD)

**Method**: the vectorized kernel natively processes 32 channels at once,
so we measured its real execution time at 32 channels for every image
size and multiplied by 2, 4, and 8 to estimate 64, 128, and 256 channels
— how long it would take to run that many 32-channel batches back to
back; for 512, 768, and 1024 channels we applied the same idea but from a
larger real measurement — 64, 96, and 128 channels respectively, matching
what one multi-column column carries at each of those totals — each again
multiplied by 8. As with the scalar section, only the multiplication is
an estimate; every base number is a real hardware measurement.

**Execution times (ms)**

| Channels | 16×16 | 64×64 | 78×78 | 114×114 | 174×174 | 366×366 |
|---:|---:|---:|---:|---:|---:|---:|
| 32   | 0.10 | 0.55  | 0.77  | 1.61  | 3.62  | 16.13  |
| 64   | 0.19 | 1.11  | 1.54  | 3.22  | 7.25  | 32.25  |
| 128  | 0.39 | 2.22  | 3.09  | 6.44  | 14.50 | 64.51  |
| 256  | 0.78 | 4.44  | 6.18  | 12.87 | 29.00 | 129.02 |
| 512  | 1.12 | 8.10  | 11.81 | 24.68 | 55.88 | —      |
| 768  | 1.26 | 11.65 | 17.37 | 35.23 | —     | —      |
| 1024 | 1.48 | 15.57 | 24.00 | —     | —     | —      |

*(— = image too large for the channel count at that row; not tested.)*

## 3. Multi-column kernel (parallel across up to 8 columns)

**Method**: unlike the previous two sections, every number below is a
direct hardware measurement — no scaling or multiplication. Channels are
distributed round-robin across columns (32 channels/column up to 8
columns, then growing channels/column up to 128 for the higher totals),
exactly as in `PERFORMANCE.md`.

**Execution times (ms)**

| Channels | 16×16 | 64×64 | 78×78 | 114×114 | 174×174 | 366×366 |
|---:|---:|---:|---:|---:|---:|---:|
| 32   | 0.08 | 0.20 | 0.27 | 0.45 | 1.03 | 4.03 |
| 64   | 0.08 | 0.20 | 0.25 | 0.45 | 0.99 | 4.02 |
| 128  | 0.08 | 0.20 | 0.25 | 0.46 | 0.97 | 4.43 |
| 256  | 0.10 | 0.22 | 0.26 | 0.48 | 1.03 | 4.85 |
| 512  | 0.10 | 0.33 | 0.48 | 0.96 | 2.51 | —    |
| 768  | 0.10 | 0.49 | 0.74 | 1.55 | —    | —    |
| 1024 | 0.11 | 0.68 | 0.94 | —    | —    | —    |

**Why growth isn't clean past 256 channels**: up to 256 total channels,
every step (32→64→...→256) adds a genuinely new, fully parallel column (32
channels/column, up to 8 columns). At 512/768/1024 channels all 8 columns
are already in use, so the extra channels can only be stacked as more
serial work *within* each column (64, then 96, then 128 channels/column) —
no longer new parallelism. That also shrinks the per-column L1 budget,
forcing the width-tiling into more, smaller slices (see "Max processable
image width vs. channels/column" in `PERFORMANCE.md`), which adds tiling
overhead on top of the extra compute. So multi-column's own time grows
slightly *faster* than the channel count past this point — which is why,
in the speedup tables below, the multi-column advantage over scalar/CPU
sometimes dips slightly (e.g. scalar-comparison at 114×114: 512ch is
1,515.8×, 768ch is only 1,409.8×) even though the NPU keeps winning in
absolute terms.

## 4. Speedups

Same grid, each cell computed from the execution times above.

**Vectorized vs. scalar**

| Channels | 16×16 | 64×64 | 78×78 | 114×114 | 174×174 | 366×366 |
|---:|---:|---:|---:|---:|---:|---:|
| 32   | 46.6×  | 99.9×  | 91.5× | 56.6× | 49.4× | 46.5× |
| 64   | 49.1×  | 99.0×  | 91.5× | 56.6× | 49.3× | 46.5× |
| 128  | 47.8×  | 99.0×  | 91.2× | 56.6× | 49.3× | 46.5× |
| 256  | 47.8×  | 99.0×  | 91.2× | 56.6× | 49.3× | 46.5× |
| 512  | 66.5×  | 108.5× | 95.4× | 59.0× | 51.2× | —     |
| 768  | 88.7×  | 113.2× | 97.3× | 62.0× | —     | —     |
| 1024 | 100.7× | 112.9× | 93.9× | —     | —     | —     |

**Why this can exceed 32×**: the vector unit is 32× wider than scalar (32
channels/cycle vs. 1), so 32× is the raw compute ceiling — yet several
cells above go well past it. That's a side effect of how the scalar number
is built (Section 1): it multiplies *one* single-channel dispatch's time
by the channel count, which effectively counts that dispatch's fixed
launch overhead once per channel, while the vector number is a single real
dispatch handling all those channels together, paying that fixed cost only
once. This inflates the ratio most at small images, where fixed overhead
is a bigger share of the total (up to 113× at 1024ch/64×64). At the
largest image (366×366), where overhead is negligible either way, the
ratio settles at a stable ~46.5× — the more honest read of the real
per-MAC efficiency gap between the two kernels.

**Multi-column vs. vectorized**

| Channels | 16×16 | 64×64 | 78×78 | 114×114 | 174×174 | 366×366 |
|---:|---:|---:|---:|---:|---:|---:|
| 32   | 1.3×  | 2.7×  | 2.9×  | 3.6×  | 3.5×  | 4.0×  |
| 64   | 2.3×  | 5.5×  | 6.1×  | 7.1×  | 7.3×  | 8.0×  |
| 128  | 4.8×  | 10.9× | 12.3× | 14.0× | 15.0× | 14.6× |
| 256  | 8.1×  | 20.2× | 23.6× | 26.6× | 28.2× | 26.6× |
| 512  | 11.6× | 24.9× | 24.4× | 25.7× | 22.3× | —     |
| 768  | 13.0× | 23.9× | 23.4× | 22.7× | —     | —     |
| 1024 | 13.2× | 23.1× | 25.5× | —     | —     | —     |

**Multi-column vs. scalar**

| Channels | 16×16 | 64×64 | 78×78 | 114×114 | 174×174 | 366×366 |
|---:|---:|---:|---:|---:|---:|---:|
| 32   | 59.1×    | 272.1×   | 264.6×   | 201.6×   | 174.3×   | 186.4× |
| 64   | 113.0×   | 542.9×   | 557.6×   | 402.0×   | 359.5×   | 372.8× |
| 128  | 227.5×   | 1,079.5× | 1,117.7× | 793.6×   | 740.1×   | 677.1× |
| 256  | 387.5×   | 2,001.6× | 2,156.4× | 1,503.2× | 1,392.6× | 1,237.1× |
| 512  | 771.3×   | 2,700.8× | 2,329.7× | 1,515.8× | 1,139.3× | —      |
| 768  | 1,153.8× | 2,703.3× | 2,272.9× | 1,409.8× | —        | —      |
| 1024 | 1,333.5× | 2,603.6× | 2,395.4× | —        | —        | —      |

## 5. Full-array NPU vs. a single CPU core

**Method**: same grid, against a real CPU baseline instead of the scalar
NPU kernel — single-threaded `torch::conv2d`, bf16, `channels_last`
(matching the NPU's HWC layout), timed directly in C++ via **LibTorch**
(PyTorch's C++ core, no Python interpreter in the timed loop) so both
sides are measured the same way, same 10 warm-up + 50 timed-run averaging
as the NPU numbers.

**Speedup (CPU / NPU, below 1× means CPU is faster)**

| Channels | 16×16 | 64×64 | 78×78 | 114×114 | 174×174 | 366×366 |
|---:|---:|---:|---:|---:|---:|---:|
| 32   | 0.28× | 0.24× | 0.24× | 0.25× | 0.23× | 0.25× |
| 64   | 0.28× | 0.39× | 0.44× | 0.45× | 0.45× | 0.41× |
| 128  | 0.23× | 0.46× | 0.53× | 0.58× | 0.73× | 1.95× |
| 256  | 0.24× | 0.80× | 0.96× | 1.19× | 2.08× | 3.27× |
| 512  | 0.36× | 1.03× | 1.10× | 1.63× | 1.88× | —     |
| 768  | 0.46× | 1.07× | 1.05× | 1.54× | —     | —     |
| 1024 | 0.50× | 1.15× | 1.18× | —     | —     | —     |

CPU wins in most of the grid; the NPU only pulls ahead with both high
channel count *and* a large image — and even there, the same
column-saturation effect noted above (Section 3) means the advantage
doesn't always keep growing with channel count (e.g. 114×114: 512ch is
1.63×, 768ch dips to 1.54×).
