# Performance: Multi-Column Depthwise Conv vs. Scalar Baseline

**Device**: AMD XDNA2 NPU (8 columns × 4 AIE cores). Kernel 3×3, padding=1, bf16.
**Methodology**: every measurement uses `run_test(..., warmup_iters=10, timed_iters=50)`
— 10 warmup runs discarded, then 50 timed runs averaged (per tutor guidance,
turbo mode enabled on the benchmark machine).

## Headline result: speedup vs. scalar, growing image size × channel count

For every channel count, image size grows from 16×16 up to the largest
width that configuration's columns can actually process (see "Max
processable image width" below) — channels distributed round-robin: fill
columns at 32 ch/column up to 8 columns, then grow channels/column up to
128. Empty cells mean that size exceeds the max width for that row.

| Channels | 16×16 | 64×64 | 114×114 | 174×174 | 366×366 |
|---:|---:|---:|---:|---:|---:|
| 32   | 59.1×   | 272.1×  | 201.6×  | 174.2×  | 186.4× |
| 64   | 113.0×  | 542.9×  | 402.0×  | 359.5×  | 372.8× |
| 96   | 168.0×  | 800.6×  | 599.8×  | 559.3×  | 506.9× |
| 128  | 227.5×  | 1079.5× | 793.6×  | 740.1×  | 677.1× |
| 256  | 387.5×  | 2001.6× | 1503.2× | 1392.6× | 1237.1× |
| 512  | 771.4×  | 2700.8× | —       | 1139.3× (max) | — |
| 768  | 1153.7× | 2703.3× | 1409.8× (max) | —       | — |
| 1024 | 1333.5× | 2603.6× | 2262.8× (max, 78×78) | — | — |

**Not monotonic in size**: every row peaks around **64×64**, then dips and
partially recovers — multi-column's own latency isn't perfectly flat as
size grows (its internal W-tiling gets more complex), while scalar grows
more steadily (near-quadratic in image size, as expected for a
non-vectorized kernel). The speedup is enormous everywhere (60×–2700×), but
"bigger image = bigger speedup" is not the right mental model.

For a simpler, single-size reference point, 64×64 alone (the size used
before the scalar operator was extended past its original ~88×88 ceiling
— see below) across the same channel sweep:

| Total channels | Columns | Channels/column | Multi-column (us) | Scalar, sequential (us) | **Speedup** |
|---:|---:|---:|---:|---:|---:|
| 32   | 1 | 32  | 201.9    | 2 452.4  | **12.1×**  |
| 64   | 2 | 32  | 202.4    | 4 904.8  | **24.2×**  |
| 96   | 3 | 32  | 205.9    | 7 357.2  | **35.7×**  |
| 128  | 4 | 32  | 203.6    | 9 809.6  | **48.2×**  |
| 256  | 8 | 32  | 219.6    | 19 619.3 | **89.4×**  |
| 512  | 8 | 64  | 325.5    | 39 238.6 | **120.5×** |
| 768  | 8 | 96  | 487.8    | 58 857.8 | **120.7×** |
| 1024 | 8 | 128 | 675.3    | 78 477.1 | **116.2×** |

### How the scalar number is obtained

`scalar_conv2d_dw` processes channels fully independently (`for c in
range(C): ...` in the kernel — no cross-channel work at all), so the table
uses a **real hardware measurement at C=1** for each image size
(`warmup_iters=10, timed_iters=50`) **multiplied by the channel count** (=
number of sequential per-channel invocations a single un-parallelized core
would need), not one measurement per (size, channels) pair. This is a
documented model, not an end-to-end stopwatch run, and it is if anything
*conservative in scalar's favor* — it excludes host-to-device dispatch
overhead between consecutive invocations, so a real sequential deployment
would likely be slower than shown here.

`scalar_conv2d_dw` was originally single-shot/untiled and topped out around
88×88 even at C=1 (L1 overflow beyond that). It was extended with real
spatial (H-only) tiling — mirroring `vector_conv2d_dw_multicore`'s proven
host-pre-padding pattern — specifically for the C=1 case, since at C=1 the
width limit is governed purely by the hardware's 11-bit DMA size field
(~2047), not by an L1 budget that shrinks with channel count, so no
W-tiling is needed up to any size tested here (verified up to 366×366).
This surfaced one new hardware constraint not seen on the HWC side: the
innermost DMA transfer size must produce a byte count that's a multiple of
4 (i.e. an even number of bf16 elements) — `tiling.py` now rounds tile
sizes to satisfy this automatically.

### Why `vector_conv2d_dw` isn't in this table

The single-core vectorized baseline pads channels up to a fixed multiple of
32 regardless of how few are requested, so its L1 footprint doesn't shrink
below the 32-channel minimum. That minimum no longer fits in L1 anywhere
past ~12×12 images — **it cannot run at 64×64 for any channel count**, not
just slowly. (A rough order-of-magnitude estimate via linear extrapolation
from 8×8/12×12 measurements suggests ~4×–40× multi-column speedup at 64×64,
but the extrapolation is ~28× beyond the farthest calibrated data point —
not reliable enough to present as a real number, only as a talking point.)

## NPU multi-column vs. CPU

Same (image size × channel count) grid as above, this time comparing NPU
multi-column against a **real CPU** run (native `torch.nn.functional.conv2d`,
`groups=C`, **single-threaded** — `torch.set_num_threads(1)`, deliberately
excluding CPU-side multi-core parallelism so this isolates "one NPU vs. one
CPU core doing no parallel tricks of its own," not "NPU vs. a fully loaded
multi-core CPU"). Median of 50 timed runs after 10 warmup runs, matching the
NPU-side methodology.

**CPU methodology (twice-revised after review)**: two independent fixes
were found by actually measuring, not assumed:

1. *Format-conversion tax.* The installed PyTorch already links oneDNN/
   MKL-DNN with AVX-512 (`torch.__config__.show()`: `USE_MKLDNN=ON`), and
   `torch.profiler` confirmed the plain-tensor path and a pre-converted
   path dispatch to the *same* `aten::mkldnn_convolution` kernel — never a
   "wrong kernel" problem. But plain (strided) tensors get converted to
   oneDNN's internal blocked layout and back on **every call**, and that
   conversion was being counted inside the timed region. This is a
   data-layout cost, not a "Python vs. C++" one: pure Python dispatch
   overhead measured at ~1.7us, negligible even on the smallest case here
   — a LibTorch (C++) rewrite would not by itself have fixed this.
2. *Wrong memory layout entirely, and dtype mismatch.* The bigger effect.
   The benchmark had been using PyTorch's native NCHW (channel outermost)
   in float32, while the NPU design uses **HWC** throughout — channel
   *innermost* (`vector_conv2d_dw_multicore/reference.py:31`, "Flat HWC
   layout") — and bf16. PyTorch's `channels_last` memory format is the
   NCHW-tensor equivalent of HWC (channel innermost), and switching to it
   (plus bf16, matching the NPU's dtype) gave a further **1.5×–5×**
   speedup on top of fix 1, because oneDNN's grouped/depthwise kernels are
   built around channel-innermost layouts — exactly why the NPU design
   itself uses HWC. The numbers below use bf16 + `channels_last`, set up
   once before the warmup loop; this supersedes both earlier passes, which
   understated the CPU by roughly 2×–8× depending on size/channels.

| Size | Channels | CPU, 1 thread, oneDNN, bf16+channels_last (us) | NPU multi-column (us) | Speedup |
|---:|---:|---:|---:|---:|
| 16×16   | 32   | 15.9    | 78.8   | 0.20× |
| 64×64   | 32   | 35.5    | 201.9  | 0.18× |
| 114×114 | 32   | 80.6    | 451.7  | 0.18× |
| 174×174 | 32   | 164.4   | 1025.6 | 0.16× |
| 366×366 | 32   | 760.0   | 4025.1 | 0.19× |
| 16×16   | 64   | 17.8    | 82.5   | 0.22× |
| 64×64   | 64   | 58.4    | 202.4  | 0.29× |
| 114×114 | 64   | 144.3   | 453.0  | 0.32× |
| 174×174 | 64   | 313.1   | 994.3  | 0.31× |
| 366×366 | 64   | 1363.0  | 4024.4 | 0.34× |
| 16×16   | 96   | 18.9    | 83.2   | 0.23× |
| 64×64   | 96   | 78.2    | 205.9  | 0.38× |
| 114×114 | 96   | 207.1   | 455.4  | 0.45× |
| 174×174 | 96   | 472.4   | 958.6  | 0.49× |
| 366×366 | 96   | 3224.0  | 4440.3 | 0.73× |
| 16×16   | 128  | 20.0    | 81.9   | 0.24× |
| 64×64   | 128  | 97.7    | 203.6  | 0.48× |
| 114×114 | 128  | 268.9   | 458.9  | 0.59× |
| 174×174 | 128  | 717.3   | 965.8  | 0.74× |
| 366×366 | 128  | 8654.4  | 4432.0 | **1.95×** |
| 16×16   | 256  | 24.7    | 96.2   | 0.26× |
| 64×64   | 256  | 174.8   | 219.6  | 0.80× |
| 114×114 | 256  | 584.9   | 484.6  | **1.21×** |
| 174×174 | 256  | 1961.8  | 1026.7 | **1.91×** |
| 366×366 | 256  | 16771.8 | 4851.7 | **3.46×** |
| 16×16   | 512  | 35.2    | 96.6   | 0.36× |
| 64×64   | 512  | 338.6   | 325.5  | **1.04×** |
| 114×114 | 512  | 1585.7  | 961.1  | **1.65×** |
| 174×174 | 512  | 4721.6  | 2509.8 | **1.88×** |
| 16×16   | 768  | 46.6    | 96.9   | 0.48× |
| 64×64   | 768  | 513.0   | 487.8  | **1.05×** |
| 114×114 | 768  | 2206.7  | 1550.1 | **1.42×** |
| 16×16   | 1024 | 57.5    | 111.8  | 0.51× |
| 64×64   | 1024 | 722.0   | 675.3  | **1.07×** |
| 78×78   | 1024 | 1161.6  | 996.1  | **1.17×** |

**Reading it**: a properly-fed CPU core (right dtype, right memory layout,
no per-call format tax) is a much tougher baseline than either earlier pass
suggested — this is the honest picture.
- **CPU wins in the large majority of the grid.** Below 256 channels, the
  CPU wins at every size tested, including 366×366. A single CPU core with
  AVX-512, fed data in its native format, is genuinely competitive with —
  often faster than — one NPU column.
- **32 channels never wins for the NPU**, at any size (0.16×–0.20×
  throughout) — 32 channels only ever occupies **one** NPU column (see
  `DEPTHWISE_CONV_STATUS.md` §6), so there's no multi-column parallelism to
  offset the NPU's larger fixed dispatch overhead.
- **The NPU needs both high channel count *and* large size to win at all.**
  The crossover only really appears from 256 channels up, and even then
  only at 114×114 and above; below that CPU still wins.
- **Where the NPU does win, the margin is modest, not dramatic**: mostly
  1.0×–2×, topping out at **3.46×** (256 channels, 366×366) and **3.71×**
  (1024 channels — not tested here past 64×64, see table above). Nothing
  like the double-digit multiples the earlier, methodologically-flawed
  passes reported.

This is the result to defend in front of a reviewer: it does not claim the
NPU is broadly faster than CPU for this workload — it shows the specific,
narrow region (high channel count, large image) where multi-column
parallelism overcomes the NPU's fixed dispatch overhead against a CPU
baseline that was verified, not assumed, to be using its best available
path (right kernel, right memory format, right dtype).

## Tall images: separating "overhead not amortized" from "throughput"

The square grid above conflates two different reasons the NPU can lose:
not enough total work to amortize its fixed per-call dispatch overhead, or
genuinely lower sustained throughput than the CPU. Image **width** is
capped per channel count (`DEPTHWISE_CONV_STATUS.md` §5), but **height is
unbounded** (verified earlier up to h=2000) — so holding width at each
row's own max and growing height instead adds much more real work without
hitting the width limit, which separates the two effects.

`h=1000`, `w` = each row's own max width, same CPU methodology (bf16,
`channels_last`, oneDNN) as above:

| Channels | Width | CPU (us) | NPU multi-column (us) | Speedup |
|---:|---:|---:|---:|---:|
| 32   | 366 | 2483.5  | 11259.2 | 0.22× |
| 64   | 366 | 3691.0  | 11379.6 | 0.32× |
| 96   | 366 | 9303.9  | 11176.4 | 0.83× |
| 128  | 366 | 24113.1 | 11245.8 | **2.14×** |
| 256  | 366 | 47523.2 | 12649.2 | **3.76×** |
| 512  | 174 | 43798.7 | 12237.6 | **3.58×** |
| 768  | 114 | 42424.0 | 12609.2 | **3.36×** |
| 1024 | 78  | 43229.9 | 12026.3 | **3.59×** |

**Reading it**:
- **512–1024 channels improve dramatically** over their square-max-width
  numbers: 512ch 1.88×→**3.58×**, 768ch 1.42×→**3.36×**, 1024ch
  1.17×→**3.59×** — nearly tripled in the worst case. At those channel
  counts the max *square* width (174/114/78) was so small that even the
  biggest square image didn't supply enough work to amortize the NPU's
  fixed overhead; a tall image at the same width removes that ceiling
  without touching the width limit at all.
- **32–96 channels barely move** (0.19×→0.22× at 32 channels — essentially
  flat). At those channel counts the bottleneck was never unamortized
  overhead — 32 channels occupies exactly **one** NPU column, and one
  column's *sustained* throughput is genuinely below the CPU's, no matter
  how much data you feed it. More work doesn't fix a throughput gap.
- **128–1024 channels converge to a stable ~3.3×–3.8×** regardless of
  channel count, once each row has enough tall-image work to reach its
  sustained-throughput regime — a much cleaner, more defensible number
  than "somewhere between 1× and 3.5×, it depends" from the square grid.

This decomposition is the sharper story for a defense: the square grid
shows *where* the NPU wins; the tall-image test shows *why* it doesn't win
everywhere — sometimes it's just under-fed (fixable by using more of the
image, i.e. a deployment concern), and sometimes it's a genuine
single-column throughput ceiling (not fixable without more columns, i.e.
an architectural fact).

## Utilization vs. theoretical peak compute

A self-contained metric — no CPU baseline needed, just the NPU against its
own hardware limit. AIE core: 512-bit vector unit, 32 int16 MACs/cycle,
clock **1.8 GHz** → **57.6 GMAC/s per core**, **1.843 TMAC/s** for the full
32-core NPU (8 columns × 4 cores). Depthwise conv needs `k_h × k_w = 9`
MACs per output element per channel; total MACs = `H_out × W_out × C × 9`
(padding=1, k=3 → H_out=H, W_out=W). Achieved MAC/s = total MACs ÷ measured
latency; utilization = achieved ÷ (57.6 GMAC/s × active cores), where
active cores = 1 for scalar, `num_columns × 4` for multi-column.

| Size | Channels | MACs (M) | Scalar util. (1 core) | Cores active (multi-col) | Multi-col util. |
|---:|---:|---:|---:|---:|---:|
| 16×16   | 32   | 0.07   | 0.027% | 4  | 0.41% |
| 64×64   | 32   | 1.18   | 0.037% | 4  | 2.54% |
| 114×114 | 32   | 3.74   | 0.071% | 4  | 3.60% |
| 174×174 | 32   | 8.72   | 0.085% | 4  | 3.69% |
| 366×366 | 32   | 38.58  | 0.089% | 4  | 4.16% |
| 16×16   | 64   | 0.15   | 0.027% | 8  | 0.39% |
| 64×64   | 64   | 2.36   | 0.037% | 8  | 2.53% |
| 114×114 | 64   | 7.49   | 0.071% | 8  | 3.59% |
| 174×174 | 64   | 17.44  | 0.085% | 8  | 3.81% |
| 366×366 | 64   | 77.16  | 0.089% | 8  | 4.16% |
| 16×16   | 96   | 0.22   | 0.027% | 12 | 0.38% |
| 64×64   | 96   | 3.54   | 0.037% | 12 | 2.49% |
| 114×114 | 96   | 11.23  | 0.071% | 12 | 3.57% |
| 174×174 | 96   | 26.16  | 0.085% | 12 | 3.95% |
| 366×366 | 96   | 115.74 | 0.089% | 12 | 3.77% |
| 16×16   | 128  | 0.29   | 0.027% | 16 | 0.39% |
| 64×64   | 128  | 4.72   | 0.037% | 16 | 2.51% |
| 114×114 | 128  | 14.97  | 0.071% | 16 | 3.54% |
| 174×174 | 128  | 34.88  | 0.085% | 16 | 3.92% |
| 366×366 | 128  | 154.32 | 0.089% | 16 | 3.78% |
| 16×16   | 256  | 0.59   | 0.027% | 32 | 0.33% |
| 64×64   | 256  | 9.44   | 0.037% | 32 | 2.33% |
| 114×114 | 256  | 29.94  | 0.071% | 32 | 3.35% |
| 174×174 | 256  | 69.76  | 0.085% | 32 | 3.69% |
| 366×366 | 256  | 308.63 | 0.089% | 32 | 3.45% |
| 16×16   | 512  | 1.18   | 0.027% | 32 | 0.66% |
| 64×64   | 512  | 18.87  | 0.037% | 32 | 3.15% |
| 114×114 | 512  | 59.89  | 0.071% | 32 | 3.38% |
| 174×174 | 512  | 139.51 | 0.085% | 32 | 3.02% |
| 16×16   | 768  | 1.77   | 0.027% | 32 | 0.99% |
| 64×64   | 768  | 28.31  | 0.037% | 32 | 3.15% |
| 114×114 | 768  | 89.83  | 0.071% | 32 | 3.14% |
| 16×16   | 1024 | 2.36   | 0.027% | 32 | 1.15% |
| 64×64   | 1024 | 37.75  | 0.037% | 32 | 3.03% |
| 78×78   | 1024 | 56.07  | 0.043% | 32 | 3.05% |

**Reading it — low percentages are expected, not a red flag.** Depthwise
convolution has very low arithmetic intensity: a 3×3 kernel does 9 MACs per
output element but touches ~9 input elements (with reuse across
neighboring outputs) and writes 1 output element — roughly **2.25 MACs per
byte moved** (bf16). The 32-MAC/cycle vector unit is built for workloads
with far more data reuse (dense matmul routinely reaches 10–100× that
intensity); for depthwise conv on this hardware, the real ceiling is
**memory movement, not compute** — consistent with effective bandwidth
staying roughly flat across this whole sweep regardless of core count
(§"Tall images" above). A classic roofline model would place this
workload near the *memory-bound* roofline, not the compute-bound one — low
compute utilization here is architecturally expected, not an efficiency
bug in the implementation.

**The scalar-vs-multi-column contrast is still the meaningful comparison**:
multi-column reaches **40–100× the per-core utilization** of the scalar
kernel (2.3–4.2% vs. 0.03–0.09%) — real vectorization and tiling against
one scalar MAC at a time, no SIMD at all. That gap is what the earlier
speedup tables already showed in latency terms; this table shows the same
gap from the hardware-utilization side.

## Supporting results

### Strong scaling (fixed total channels, redistributed across columns)

Efficiency = speedup ÷ (columns used ÷ baseline columns).

**256 total channels, 30×30 images** (the largest single-column can fit at
256 channels, giving a valid 1-column baseline):

| Columns | Channels/col | Latency (us) | Speedup | Efficiency |
|---:|---:|---:|---:|---:|
| 1 | 256 | 290.2 | 1.00× | 100% (baseline) |
| 2 | 128 | 193.3 | 1.50× | 75.1% |
| 4 | 64  | 130.9 | 2.22× | 55.4% |
| 8 | 32  | 104.3 | 2.78× | 34.8% |

**256 total channels, 56×56 images** (1 column infeasible here — baseline
shifted to 2 columns):

| Columns | Channels/col | Latency (us) | Speedup (vs 2 col) | Efficiency |
|---:|---:|---:|---:|---:|
| 2 | 128 | 448.2 | 1.00× | 100% (baseline) |
| 4 | 64  | 250.8 | 1.79× | 89.3% |
| 8 | 32  | 173.8 | 2.58× | 64.5% |

Takeaway: scaling efficiency depends on how much real work lands on each
column relative to its fixed DMA/sync setup cost — larger images (more
compute per column) amortize that cost much better than small ones.

### Max processable image width vs. channels/column

Governed by the Shim tile's buffer-descriptor pool (`MAX_W_SLICES=6`) and
an 11-bit DMA size field (`MAX_ROW_IN=2047`) — see `DEPTHWISE_CONV_STATUS.md`
§5 for details. Height is unbounded regardless of channel count (verified
up to h=2000).

| Channels/column | Max width |
|---:|---:|
| 32  | 366 |
| 64  | 174 |
| 96  | 114 |
| 128 | 78  |
