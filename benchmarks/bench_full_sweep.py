#!/usr/bin/env python3
"""Full benchmark sweep: scalar vs vector (single-core) vs multicolumn
(parallel across up to 8 columns), for total channel counts from 32 to 1024,
plus max-width-vs-channels confirmation. Scratch exploration script."""

import sys
import torch
import aie.utils as aie_utils

from iron.operators.scalar_conv2d_dw.op import ScalarConv2DDW
from iron.operators.scalar_conv2d_dw.reference import generate_golden_reference as scalar_ref
from iron.operators.vector_conv2d_dw.op import VectorConv2DDW
from iron.operators.vector_conv2d_dw.reference import generate_golden_reference as vector_ref
from iron.operators.vector_conv2d_dw_multicolumn.op import VectorConv2DDWMColumn
from iron.operators.vector_conv2d_dw_multicore.reference import generate_golden_reference as mc_ref
from iron.operators.vector_conv2d_dw_multicolumn.test import pad_channels, build_padded_input, depthwise_valid
from iron.common.test_utils import run_test
from iron.common.context import AIEContext

H = W = 8
K_H = K_W = 3
PADDING = 1


def run_scalar(c):
    ctx = AIEContext()
    try:
        golden_ref = scalar_ref(c, H, W, K_H, K_W, padding=PADDING)
        operator = ScalarConv2DDW(c=c, h=H, w=W, k_h=K_H, k_w=K_W, padding=PADDING, context=ctx)
        weights = golden_ref["Kernel"].reshape(-1)
        pad = operator.weight_size - weights.numel()
        if pad:
            weights = torch.nn.functional.pad(weights, (0, pad))
        input_buffers = {"input": golden_ref["Input"], "weights": weights}
        output_buffers = {"output": golden_ref["Output"]}
        errors, latency_us, bw = run_test(
            operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-6,
            warmup_iters=10, timed_iters=50,
        )
        return None if errors else latency_us
    except Exception as e:
        return f"EXC:{type(e).__name__}"
    finally:
        aie_utils.DefaultNPURuntime.cleanup()


def run_vector(c):
    ctx = AIEContext()
    try:
        golden_ref = vector_ref(H, W, c, K_H, K_W, padding=PADDING)
        operator = VectorConv2DDW(h=H, w=W, c=c, k_h=K_H, k_w=K_W, padding=PADDING, context=ctx)
        cp = operator.channels_padded
        input_buffers = {
            "input": pad_channels(golden_ref["Input"], cp),
            "weights": pad_channels(golden_ref["Kernel"], cp).reshape(-1),
        }
        output_buffers = {"output": pad_channels(golden_ref["Output"], cp)}
        errors, latency_us, bw = run_test(
            operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-6,
            warmup_iters=10, timed_iters=50,
        )
        return None if errors else latency_us
    except Exception as e:
        return f"EXC:{type(e).__name__}"
    finally:
        aie_utils.DefaultNPURuntime.cleanup()


def run_multicolumn(c_per_col, num_columns, h=H, w=W):
    ctx = AIEContext()
    try:
        golden_ref = mc_ref(h, w, c_per_col * num_columns, K_H, K_W, padding=PADDING)
        operator = VectorConv2DDWMColumn(
            h=h, w=w, c=c_per_col, k_h=K_H, k_w=K_W, padding=PADDING,
            num_cores=4, num_columns=num_columns, context=ctx,
        )
        g = operator.tiling
        cp = g["c_padded"]
        in_blocks, w_blocks, out_blocks = [], [], []
        for col in range(num_columns):
            x_slice = golden_ref["Input"][:, :, col * c_per_col:(col + 1) * c_per_col]
            w_slice = golden_ref["Kernel"][:, :, col * c_per_col:(col + 1) * c_per_col]
            x_cpadded = pad_channels(x_slice, cp)
            weight_cpadded = pad_channels(w_slice, cp)
            x_padded = build_padded_input(x_cpadded, g["hp_padded"], g["wp_padded"], cp, PADDING)
            expected = depthwise_valid(x_padded, weight_cpadded)
            in_blocks.append(x_padded.reshape(-1))
            w_blocks.append(weight_cpadded.reshape(-1))
            out_blocks.append(expected.reshape(-1))
        input_buffers = {"input": torch.cat(in_blocks), "weights": torch.cat(w_blocks)}
        output_buffers = {"output": torch.cat(out_blocks)}
        errors, latency_us, bw = run_test(
            operator, input_buffers, output_buffers, rel_tol=0.04, abs_tol=1e-6,
            warmup_iters=10, timed_iters=50,
        )
        return None if errors else latency_us
    except Exception as e:
        return f"EXC:{type(e).__name__}: {e}"
    finally:
        aie_utils.DefaultNPURuntime.cleanup()


def main():
    # (total_c, num_columns, c_per_col) -- fills columns with 32 ch/col first
    # (1..8 columns => 32..256 total), then grows channels/column once all 8
    # columns are active (256..1024 total). Matches the round-robin design plan.
    points = [
        (32, 1, 32),
        (64, 2, 32),
        (96, 3, 32),
        (128, 4, 32),
        (256, 8, 32),
        (512, 8, 64),
        (768, 8, 96),
        (1024, 8, 128),
    ]

    print(f"=== {H}x{W} spatial, kernel {K_H}x{K_W}, padding={PADDING} ===")
    print(f"{'total_c':>8} {'cols':>4} {'c/col':>6} {'scalar_us':>12} {'vector_us':>12} {'multicol_us':>13} {'spd_v_scalar':>13} {'spd_v_vector':>13}")
    for total_c, num_columns, c_per_col in points:
        mc_lat = run_multicolumn(c_per_col, num_columns)
        sys.stdout.flush()

        if total_c <= 96:
            sc_lat = run_scalar(total_c)
            vc_lat = run_vector(total_c)
        else:
            sc_lat = "N/A(L1)"
            vc_lat = "N/A(L1)"
        sys.stdout.flush()

        def fmt(x):
            return f"{x:.1f}" if isinstance(x, float) else str(x)

        def speedup(base, mc):
            if isinstance(base, float) and isinstance(mc, float) and mc > 0:
                return f"{base / mc:.2f}x"
            return "N/A"

        print(f"{total_c:>8} {num_columns:>4} {c_per_col:>6} {fmt(sc_lat):>12} {fmt(vc_lat):>12} {fmt(mc_lat):>13} "
              f"{speedup(sc_lat, mc_lat):>13} {speedup(vc_lat, mc_lat):>13}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
