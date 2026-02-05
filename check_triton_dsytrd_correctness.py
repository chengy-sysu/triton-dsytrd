#!/usr/bin/env python3
"""
Correctness check for triton-dsytrd: compare eigvals of A vs eigvals of tridiagonal T.
"""
from __future__ import annotations

import argparse

import torch

import triton_dsytrd_run as triton_dsytrd
import triton_tridiag_2stage as t2


def tridiag_to_dense(d: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    n = d.numel()
    T = torch.zeros((n, n), device=d.device, dtype=d.dtype)
    T.diagonal().copy_(d)
    idx = torch.arange(n - 1, device=d.device)
    T[idx + 1, idx] = e
    T[idx, idx + 1] = e
    return T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
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

    A = torch.empty((args.n, args.n), device="cuda", dtype=torch.float64)
    triton_dsytrd._init_sym_inplace_(A, args.seed, triton_dsytrd.symmetrize_upper_from_lower_kernel)
    A_ref = A.clone()
    torch.cuda.synchronize()

    t2.sbr_to_band_dbbr_triton_(A, args.b, update="gemm1", symmetrize=False)
    d_tr, e_tr = triton_dsytrd.bc_band_to_tridiag_pipeline_gpu_(A, args.b, workers=workers)
    torch.cuda.synchronize()

    T_tr = tridiag_to_dense(d_tr, e_tr)
    w_A = torch.linalg.eigvalsh(A_ref)
    w_T = torch.linalg.eigvalsh(T_tr)
    diff = (w_A - w_T).abs()
    max_abs = diff.max().item()
    denom = w_A.abs().max().item()
    rel = max_abs / (denom if denom != 0.0 else 1.0)
    print(f"eigval compare (A vs triton_T): max_abs={max_abs:.3e} rel={rel:.3e}")


if __name__ == "__main__":
    main()
