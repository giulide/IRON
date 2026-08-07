#!/usr/bin/env python3
"""CPU (single-threaded, native PyTorch conv2d) vs NPU multi-column, for the
same (h, w, total_channels) grid already used for the scalar-vs-multicolumn
comparison. Scratch exploration script."""

import os
import sys
import time
import torch

torch.set_num_threads(1)  # single-threaded CPU baseline (no CPU-side
                          # parallelism -- established convention this
                          # session: multi-threaded CPU would be "barare")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_full_sweep import run_multicolumn


def run_cpu(h, w, c, k_h=3, k_w=3, padding=1, warmup=10, timed=50):
    """CPU baseline using PyTorch's oneDNN (MKL-DNN) backend, bf16 (matching
    the NPU's dtype throughout this project) and PyTorch's `channels_last`
    memory format (the NCHW-tensor equivalent of NPU's HWC layout: channel
    innermost) -- both set up ONCE before timing starts, not per call.

    Two corrections layered on top of each other, found by benchmarking
    this exact grid, not assumed:
    1. Plain CHW (PyTorch's native NCHW) tensors made every call pay a
       real format-conversion cost converting to/from oneDNN's internal
       blocked layout (confirmed via torch.profiler: both the plain and
       the pre-converted path dispatch to the identical
       `aten::mkldnn_convolution` kernel -- the plain path is just paying
       a repeated conversion tax the pre-converted path doesn't).
    2. Independently, and a much bigger effect, `channels_last` memory
       format is dramatically faster than CHW for this workload (up to
       ~5x at some sizes) -- oneDNN's grouped/depthwise convolution
       kernels are optimized around channel-innermost layouts, matching
       why the NPU design itself uses HWC.
    """
    torch.manual_seed(0)
    x = torch.randn(1, c, h, w, dtype=torch.bfloat16).to(memory_format=torch.channels_last)
    weight = torch.randn(c, 1, k_h, k_w, dtype=torch.bfloat16)
    for _ in range(warmup):
        torch.nn.functional.conv2d(x, weight, stride=1, padding=padding, groups=c)
    times = []
    for _ in range(timed):
        t0 = time.perf_counter()
        torch.nn.functional.conv2d(x, weight, stride=1, padding=padding, groups=c)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1e6  # median, us


# (total_c, num_columns, c_per_col, [sizes]) -- same grid as the
# scalar-vs-multicolumn comparison, bounded by each row's own max width.
ROWS = [
    (32, 1, 32, [16, 64, 114, 174, 366]),
    (64, 2, 32, [16, 64, 114, 174, 366]),
    (96, 3, 32, [16, 64, 114, 174, 366]),
    (128, 4, 32, [16, 64, 114, 174, 366]),
    (256, 8, 32, [16, 64, 114, 174, 366]),
    (512, 8, 64, [16, 64, 114, 174]),
    (768, 8, 96, [16, 64, 114]),
    (1024, 8, 128, [16, 64, 78]),
]

# Reuse already-measured multicolumn latencies (us) from this session.
KNOWN_MC = {
    (16, 32): 78.79, (16, 64): 82.47, (16, 96): 83.18, (16, 128): 81.90,
    (16, 256): 96.16, (16, 512): 96.61, (16, 768): 96.89, (16, 1024): 111.77,
    (64, 32): 201.9, (64, 64): 202.4, (64, 96): 205.9, (64, 128): 203.6,
    (64, 256): 219.6, (64, 512): 325.5, (64, 768): 487.8, (64, 1024): 675.3,
    (78, 1024): 996.10,
    (114, 32): 451.73, (114, 64): 452.98, (114, 96): 455.39, (114, 128): 458.95,
    (114, 256): 484.58, (114, 512): 961.10, (114, 768): 1550.10,
    (174, 32): 1025.63, (174, 64): 994.32, (174, 96): 958.58, (174, 128): 965.84,
    (174, 256): 1026.66, (174, 512): 2509.85,
    (366, 32): 4025.06, (366, 64): 4024.42, (366, 96): 4440.30, (366, 128): 4431.99,
    (366, 256): 4851.72,
}


def main():
    print(f"{'size':>10} {'channels':>9} {'cpu_us':>12} {'npu_us':>12} {'speedup':>10}")
    for total_c, num_columns, c_per_col, sizes in ROWS:
        for s in sizes:
            cpu_us = run_cpu(s, s, total_c)
            npu_us = KNOWN_MC.get((s, total_c))
            if npu_us is None:
                npu_us = run_multicolumn(c_per_col, num_columns, h=s, w=s)
            speedup = cpu_us / npu_us if isinstance(npu_us, float) else None
            spd_str = f"{speedup:.2f}x" if speedup is not None else "N/A"
            print(f"{s:>4}x{s:<5} {total_c:>9} {cpu_us:>12.1f} {npu_us:>12.1f} {spd_str:>10}")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
