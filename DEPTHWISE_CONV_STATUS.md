# Depthwise Conv2D on AMD XDNA2 NPU — Technical Status

**Date**: 2026-08-02
**Branch**: `socdaml-giulio` (base `devel`)
**Status**: Multi-column (up to 8 columns / 32 cores) depthwise conv working, tested, and benchmarked.

## 1. Target hardware

AMD XDNA2 NPU: 8 columns × 6 rows. Each column = 1 Shim tile (row 0, DRAM
I/O, native budget 2 DMA-in + 2 DMA-out channels) + 1 Mem tile (row 1, 512KB
scratchpad, its own separate DMA-channel budget) + 4 AIE compute tiles (rows
2–5, 512-bit vector unit / 32 int16 MACs per cycle, 64KB L1 each).

## 2. Implementation progression

`naive → scalar_conv2d_dw → vector_conv2d_dw → vector_conv2d_dw_multicore (1 column, 4 cores) → vector_conv2d_dw_multicolumn (up to 8 columns, 32 cores)`

Each stage is a separate operator under `iron/operators/`. `conv2d_dw` (the
very first attempt) was never finished/tested — not a reuse target.
`scalar_conv2d_dw` and `vector_conv2d_dw` are **single-core, untiled**: one
DMA transfer moves the *entire* padded image in one shot, so they only work
while `h*w*c_padded` fits in a single core's 64KB L1 (see §5).

## 3. Single-column design — `vector_conv2d_dw_multicore/`

- **`tiling.py`**: pure-Python geometry (`compute_tiling`), no AIE imports.
  Computes padded shapes, W-tiling (`num_w_sub`), and per-core row
  distribution. Owns the two hardware-limit constants (§5): `MAX_ROW_IN`,
  `MAX_W_SLICES`.
- **`design.py`**: builds the IRON graph. Key pieces:
  - `_check_hw_limits(g, col, w, c)` — raises a clear `ValueError` for a
    hardware-limit violation. **Must run before `rt.sequence()`/
    `rt.task_group()` are opened** — raising once a task group is open gets
    masked by an unrelated "Failed to close task groups" error from the
    `with` block's cleanup.
  - `_make_conv_kernel(g, k_h, k_w)` — builds the single `Kernel(...)`
    object shared by every core/column that uses it.
  - `_build_column(rt, tg, col, ...)` — the reusable unit: one column's 4
    Workers + MemTile, wired via `split()`/`join()`, with **every tile
    explicitly pinned** (`Tile(col,1)` MemTile, `Tile(col,2..5)` cores,
    `Tile(col,0)` Shim fill/drain — see §4 for why this must be explicit).
  - `conv2d_dw_multicore(dev, h, w, c, k_h, k_w, padding, num_cores)` —
    public entry point, now a thin wrapper: opens one `Runtime`, calls
    `_build_column` once with `col=0`.
  - Data-movement trick: output rows are assigned round-robin
    (`core = tile_index % num_cores`) instead of contiguous bands, so a
    *single* strided Shim↔MemTile transfer covers every core's tiles →
    only 3 Shim channels used per column (in: data + weights, out: data),
    inside the native 2-in/2-out budget. Width beyond what one L1 tile
    fits is handled by W-tiling: a Python loop issuing one Shim↔MemTile
    transfer pair per W-slice (bounded by `MAX_W_SLICES`, §5).
- **`op.py`**: `VectorConv2DDWMC` dataclass (arg spec, kernel artifact).
- **`reference.py`** / **`test.py`**: PyTorch golden reference + 125
  passing pytest cases (spatial sizes, W-tiling, channel counts 32/64/96,
  and explicit guard-boundary tests).

## 4. Multi-column extension — `vector_conv2d_dw_multicolumn/`

Depthwise conv has no cross-channel dependency, so column `col` computes
the full `h×w` extent for its own disjoint slice of `c` channels — **`c`
means channels *per column***; total channels handled = `c * num_columns`.
`design.py`'s `conv2d_dw_multicolumn()` imports `_build_column` /
`_check_hw_limits` / `_make_conv_kernel` unchanged and calls
`_build_column` once per column inside **one shared** `Runtime`/
`task_group` (so every column's DMA runs concurrently, not serialized).
The 3 top-level tensors are `num_columns` independently-padded HWC blocks
stacked back-to-back (**not** a literal interleaved `[h,w,c*num_columns]`
tensor) — offset `col * per_col_size` into the flat buffer.

### Two real bugs found while extending to multiple columns

1. **Guard-masking** (Python-level): described in §3 — fixed by moving
   `_check_hw_limits` to run before `rt.sequence()` opens.
2. **MLIR "redefinition of symbol"**: constructing a fresh `Kernel(...)`
   object per column (even with identical name/signature) makes MLIR
   verification fail, because each `Kernel()` call emits its own symbol
   declaration. Fixed by constructing the kernel object once
   (`_make_conv_kernel`) and sharing it across every column's Workers.
3. **Shim mis-routing risk** (found via source inspection of
   `SequentialPlacer`, confirmed by design, not hit as a live bug):
   `SequentialPlacer` places all Workers first via a flat column-major
   counter that ignores ObjectFifo connectivity, and separately resolves
   *unpinned* Shim endpoints via a "common column" heuristic that only
   sees an ObjectFifo's **direct** endpoints — for the MemTile-routed
   fifos (`split()`/`join()`) that never includes the downstream compute
   tiles, so every column's Shim traffic would silently resolve toward
   column 0 and spill from there once oversubscribed. Fixed by pinning
   `placement=Tile(col, 0)` explicitly on every `rt.fill()`/`rt.drain()`
   call, not relying on inference. A dedicated regression case at
   `num_columns=3` exists specifically because this failure mode is
   invisible at 1–2 columns (column 0 has spare budget to absorb a 2nd
   claimant by luck) and only becomes a loud failure at 3+.

## 5. Known limitations (verified both analytically and on real hardware)

All limits below are **per column**, independent of `num_columns` — each
column has its own independent Shim tile and MemTile budget.

- **`MAX_ROW_IN = 2047`**: the innermost Shim DMA transfer dimension
  (`w_in_tile * c_padded`) is an 11-bit hardware field that silently wraps
  to 0 at exactly 2048. Not software-fixable — genuine silicon overflow.
- **`MAX_W_SLICES = 6`**: the Shim buffer-descriptor pool supports at most
  6 W-slice transfer pairs end-to-end. 7 slices compiles but **hangs at
  execution** (`ERT_CMD_STATE_TIMEOUT`); 8 fails to compile. Confirmed by
  deliberately bypassing the guard on real hardware (single column, to
  limit blast radius on this shared machine): reproduced the documented
  timeout at `c=128`, matching the original `c=32` discovery — the limit
  held across configurations, so it's very likely a true fixed hardware
  resource (BD-ID pool size), not a fluke of one channel count. XRT's own
  command timeout fired and raised a catchable Python exception (not an
  unrecoverable hang); the device was confirmed healthy immediately after
  via a normal test run. Not yet investigated whether smarter DMA
  descriptor packing (avoiding the `shim_dma_single_bd_task` repeat_count
  workaround's wasted dimension slot — see `design.py` module docstring)
  could raise this; would be a nontrivial, uncertain-payoff investigation.

  **Max image width per channels/column** (height is unaffected — see
  below):

  | channels/column | max W | fails at |
  |---|---|---|
  | 32  | 366 | 367 |
  | 64  | 174 | 175 |
  | 96  | 114 | 115 |
  | 128 |  78 |  79 |

  Exceeding the limit raises a clear `ValueError` at operator construction
  time, before touching hardware.

- **Image height is unbounded** at any channels/column (verified up to
  h=2000, including combined with max-width w=78 at c=128/column) — the
  round-robin trick turns "which core" into one linear DMA sweep with no
  height term in any guard.
- **A single column tops out around 256–384 total channels** before
  `MAX_W_SLICES` forces multi-column distribution (at `w=8`, single-column
  `c=512` already fails: `needs 8 W-slices`). Beyond that, spreading
  channels across columns isn't just faster, it's the only way to run.
- **`scalar_conv2d_dw`/`vector_conv2d_dw` (single-core, untiled) L1
  ceiling** (at h=w=8): scalar tops out at c=117, vector at c=96 — the
  whole padded image must fit in one core's 64KB L1 in a single shot, no
  tiling exists in these designs.

## 6. Performance (h=w=8, kernel 3×3, padding=1, bf16)

Compared 4 levels: scalar (1 core, no vectorization) → vector (1 core,
SIMD) → single-column multicore (4 cores) → multi-column (up to 32 cores).
Scalar/vector numbers beyond their L1 ceiling are the *sequential-chunked*
model (repeatedly call the max-safe-size kernel and sum latency — a
realistic proxy for "how long would a single un-tiled core take run
back-to-back", not a real single-shot measurement).

| total channels | scalar (us) | vector (us) | multicore/1 col (us) | multicolumn (us) |
|---|---|---|---|---|
| 32   | 627.1    | 88.6   | 62.4 | 67.2 |
| 64   | 1246.7   | 108.7  | 82.0 | 82.8 |
| 96   | 1804.4   | 115.1  | 69.8 | 80.9 |
| 128  | 4329.1   | 231.2  | 90.7 | 66.5 |
| 256  | 6493.6   | 346.8  | 79.8 | 73.5 |
| 512  | 10822.7  | 693.5  | infeasible (`MAX_W_SLICES`) | 76.8 |
| 768  | 15151.8  | 924.7  | infeasible | 73.8 |
| 1024 | 19480.9  | 1271.4 | infeasible | 74.6 |

**Reading it**: speedup vs. scalar grows from **9.3×** (32 ch) to **261×**
(1024 ch); vs. vector from **1.3×** to **17.0×**. It is *not* constant —
each parallelism level (SIMD → 4 cores → 8 columns) has a fixed
data-movement/synchronization overhead that needs enough work to amortize:
going from 1 core to 4 (one column) barely helps at 32–96 channels (the
MemTile split/join overhead ≈ the parallelism gain at that scale) and only
pays off clearly from ~128 channels up; going from 1 column to 8 similarly
shows near-zero (sometimes slightly negative, within measurement noise)
benefit at 32–96 channels, becoming worthwhile at 128+ and *mandatory*
past ~256–384 (single column can no longer fit the transfer count).

**Parallelism verified real, not just configured**: holding per-column
work fixed (h=64,w=64,c=128) and varying only `num_columns`, latency stays
~535–540us for 1/2/4 columns and rises only ~19% at 8 (likely shared
DRAM/NoC bandwidth, not serialization) — a genuinely sequential execution
would show ~8× growth. Cross-checked against the generated MLIR: each
column's Workers/MemTile/Shim land on physically distinct, non-colliding
tiles.

## 7. Open / not yet explored

- **Channel-per-*core*** (not per-column) parallelization for total
  channel counts beyond 1024 (target discussed: up to 2688) — assign 32
  channels directly to each of the 32 cores rather than splitting first by
  column then tiling spatially within a column. Not yet designed or
  investigated; the max-width-per-channel-count relationship in §5 would
  presumably still apply per-core rather than per-column, but this hasn't
  been derived.
- Whether `MAX_W_SLICES` can be raised via smarter DMA descriptor packing
  (see §5) — deliberately deprioritized pending confirmation this is
  actually a practical blocker for real MobileNet/YOLO layer shapes.

## 8. Environment setup

```bash
cd /scratch/gguerrieri/IRON
export MAMBA_ROOT_PREFIX=/scratch/gguerrieri/micromamba_root
eval "$(/scratch/gguerrieri/bin/micromamba shell hook -s bash)"
micromamba activate iron-env

# Single-column regression suite
python -m pytest iron/operators/vector_conv2d_dw_multicore/test.py -q

# Multi-column suite (1 to 8 columns)
python -m pytest iron/operators/vector_conv2d_dw_multicolumn/test.py -q -m extensive
```

Shared machine: keep heavy files under `/scratch/gguerrieri` (not home),
cap parallel build jobs at `-j8`, and never touch other users' files.
