#!/usr/bin/env python3
"""
Benchmark triton-dsytrd over a fixed set of sizes and report TFLOP/s (model=4/3*n^3).
"""
from __future__ import annotations

import argparse
import time

import torch

import triton_dsytrd_run as triton_dsytrd
import triton_tridiag_2stage as t2


SIZES = [4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 36864, 40960, 45056]


def eff_tflops_sytrd(n: int, ms: float) -> float:
    flops = (4.0 / 3.0) * (float(n) ** 3)
    return flops / (ms * 1e-3) / 1e12


def run_once(A: torch.Tensor, b: int, workers: int | None, seed: int) -> tuple[float, float, float]:
    triton_dsytrd._init_sym_inplace_(A, seed, triton_dsytrd.symmetrize_upper_from_lower_kernel)
    torch.cuda.synchronize()

    t0 = time.time()
    t2.sbr_to_band_dbbr_triton_(A, b, update="gemm1", symmetrize=False)
    torch.cuda.synchronize()
    t1 = time.time()
    _ = triton_dsytrd.bc_band_to_tridiag_pipeline_gpu_(A, b, workers=workers)
    torch.cuda.synchronize()
    t2_end = time.time()
    return (t1 - t0) * 1e3, (t2_end - t1) * 1e3, (t2_end - t0) * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0, help="0 => auto (SM count)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    if args.b > 64:
        raise SystemExit("bc-pipeline path currently assumes b<=64")

    torch.set_default_dtype(torch.float64)
    workers = None if args.workers == 0 else args.workers

    print("n,b,sbr_ms,stage2_ms,total_ms,TFLOP/s(model=4/3*n^3)")
    for n in SIZES:
        A = torch.empty((n, n), device="cuda", dtype=torch.float64)
        # warm once
        _ = run_once(A, args.b, workers, args.seed)
        torch.cuda.empty_cache()
        # timed once
        s1_ms, s2_ms, total_ms = run_once(A, args.b, workers, args.seed + 1)
        tf = eff_tflops_sytrd(n, total_ms)
        print(f"{n},{args.b},{s1_ms:.3f},{s2_ms:.3f},{total_ms:.3f},{tf:.3f}")


if __name__ == "__main__":
    main()
