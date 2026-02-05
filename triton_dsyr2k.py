import argparse
import time

import torch
import triton
import triton.language as tl


@triton.jit
def dsyr2k_tt_outer_kernel(
    C_ptr, A_ptr, B_ptr,
    N,  # C is NxN
    stride_c0, stride_c1,
    stride_a0, stride_a1,  # A is KxN
    stride_b0, stride_b1,  # B is KxN
    alpha: tl.constexpr,
    beta: tl.constexpr,
    UPLO: tl.constexpr,  # 0=lower, 1=upper
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """
    DSYR2K (trans='T') in-place on triangular part of C:
      C := alpha * (A^T B + B^T A) + beta * C
    where A,B are KxN and C is NxN symmetric; we only update uplo triangle.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Skip tiles completely outside the requested triangle.
    # NOTE: Must use element indices (BM/BN can be different); comparing pid_m/pid_n is incorrect.
    m0 = pid_m * BM
    n0 = pid_n * BN
    if UPLO == 0:
        # lower: skip if the entire tile is strictly above the diagonal
        if (m0 + (BM - 1)) < n0:
            return
    else:
        # upper: skip if the entire tile is strictly below the diagonal
        if m0 > (n0 + (BN - 1)):
            return

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    mask_m = offs_m < N
    mask_n = offs_n < N
    mask_tile = mask_m[:, None] & mask_n[None, :]

    # Triangular mask inside the diagonal tile.
    if UPLO == 0:
        mask_tri = offs_m[:, None] >= offs_n[None, :]
    else:
        mask_tri = offs_m[:, None] <= offs_n[None, :]
    mask = mask_tile & mask_tri

    # beta specialization: for beta==0 we can avoid reading C entirely.
    if beta == 0.0:
        c = tl.zeros((BM, BN), dtype=tl.float64)
    else:
        c = tl.load(C_ptr + offs_m[:, None] * stride_c0 + offs_n[None, :] * stride_c1,
                    mask=mask, other=0.0).to(tl.float64)
        if beta != 1.0:
            c = c * beta

    acc = tl.zeros((BM, BN), dtype=tl.float64)

    # K is typically small (bandwidth/panel size). Keep it static for unrolling.
    # A,B are KxN row-major contiguous in the intended use (stride_*1 == 1), which makes
    # per-k vector loads coalesced for the trailing-matrix update use case.
    for kk in tl.static_range(0, K):
        a_m = tl.load(A_ptr + kk * stride_a0 + offs_m * stride_a1, mask=mask_m, other=0.0).to(tl.float64)
        a_n = tl.load(A_ptr + kk * stride_a0 + offs_n * stride_a1, mask=mask_n, other=0.0).to(tl.float64)
        b_m = tl.load(B_ptr + kk * stride_b0 + offs_m * stride_b1, mask=mask_m, other=0.0).to(tl.float64)
        b_n = tl.load(B_ptr + kk * stride_b0 + offs_n * stride_b1, mask=mask_n, other=0.0).to(tl.float64)
        # Outer products for the two GEMM-like terms.
        acc += a_m[:, None] * b_n[None, :] + b_m[:, None] * a_n[None, :]

    c = c + (alpha * acc)
    tl.store(C_ptr + offs_m[:, None] * stride_c0 + offs_n[None, :] * stride_c1, c, mask=mask)


@triton.jit
def dsyr2k_tt_dot_kernel(
    C_ptr, A_ptr, B_ptr,
    N,  # C is NxN
    stride_c0, stride_c1,
    stride_a0, stride_a1,  # A is KxN
    stride_b0, stride_b1,  # B is KxN
    alpha: tl.constexpr,
    beta: tl.constexpr,
    UPLO: tl.constexpr,  # 0=lower, 1=upper
    K: tl.constexpr,
    BK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """
    Same math as dsyr2k_tt_outer_kernel, but uses block-K tl.dot to avoid
    fully unrolling K outer products (helps both compile-time and, on newer
    Triton versions, can generate better code).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m0 = pid_m * BM
    n0 = pid_n * BN
    if UPLO == 0:
        if (m0 + (BM - 1)) < n0:
            return
    else:
        if m0 > (n0 + (BN - 1)):
            return

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    mask_m = offs_m < N
    mask_n = offs_n < N
    mask_tile = mask_m[:, None] & mask_n[None, :]

    if UPLO == 0:
        mask_tri = offs_m[:, None] >= offs_n[None, :]
    else:
        mask_tri = offs_m[:, None] <= offs_n[None, :]
    mask = mask_tile & mask_tri

    if beta == 0.0:
        c = tl.zeros((BM, BN), dtype=tl.float64)
    else:
        c = tl.load(C_ptr + offs_m[:, None] * stride_c0 + offs_n[None, :] * stride_c1,
                    mask=mask, other=0.0).to(tl.float64)
        if beta != 1.0:
            c = c * beta

    acc = tl.zeros((BM, BN), dtype=tl.float64)

    # Contract over K in BK-sized chunks:
    #   acc += A_mT @ B_n + B_mT @ A_n
    # where A_mT/B_mT are (BM,BK) and A_n/B_n are (BK,BN).
    for k0 in tl.static_range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        mask_k = offs_k < K

        A_mT = tl.load(
            A_ptr + offs_k[None, :] * stride_a0 + offs_m[:, None] * stride_a1,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float64)
        B_mT = tl.load(
            B_ptr + offs_k[None, :] * stride_b0 + offs_m[:, None] * stride_b1,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float64)
        A_n = tl.load(
            A_ptr + offs_k[:, None] * stride_a0 + offs_n[None, :] * stride_a1,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float64)
        B_n = tl.load(
            B_ptr + offs_k[:, None] * stride_b0 + offs_n[None, :] * stride_b1,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float64)

        acc += tl.dot(A_mT, B_n).to(tl.float64)
        acc += tl.dot(B_mT, A_n).to(tl.float64)

    c = c + (alpha * acc)
    tl.store(C_ptr + offs_m[:, None] * stride_c0 + offs_n[None, :] * stride_c1, c, mask=mask)


@triton.jit
def tri_scale_kernel(
    C_ptr,
    N,
    stride_c0, stride_c1,
    beta: tl.constexpr,
    UPLO: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """Scale (or zero) the requested triangle of C in-place: C := beta * C (triangle only)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m0 = pid_m * BM
    n0 = pid_n * BN
    if UPLO == 0:
        if (m0 + (BM - 1)) < n0:
            return
    else:
        if m0 > (n0 + (BN - 1)):
            return

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_m = offs_m < N
    mask_n = offs_n < N
    mask_tile = mask_m[:, None] & mask_n[None, :]

    if UPLO == 0:
        mask_tri = offs_m[:, None] >= offs_n[None, :]
    else:
        mask_tri = offs_m[:, None] <= offs_n[None, :]
    mask = mask_tile & mask_tri

    if beta == 0.0:
        c = tl.zeros((BM, BN), dtype=tl.float64)
    else:
        c = tl.load(C_ptr + offs_m[:, None] * stride_c0 + offs_n[None, :] * stride_c1,
                    mask=mask, other=0.0).to(tl.float64)
        if beta != 1.0:
            c = c * beta

    tl.store(C_ptr + offs_m[:, None] * stride_c0 + offs_n[None, :] * stride_c1, c, mask=mask)


@triton.jit
def dsyr2k_tt_dotk_kernel(
    C_ptr, A_ptr, B_ptr,
    N,  # C is NxN
    K,  # A,B are KxN (runtime)
    stride_c0, stride_c1,
    stride_a0, stride_a1,  # A is KxN
    stride_b0, stride_b1,  # B is KxN
    alpha: tl.constexpr,
    UPLO: tl.constexpr,  # 0=lower, 1=upper
    BK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """
    K-parallel dsyr2k update using atomic adds:
      C := C + alpha*(A^T B + B^T A)
    where each program computes one BK-slice of K and atomically accumulates into C.

    This avoids compile-time unrolling over K (good for very large K), but needs
    a prior scaling step on C if beta != 1 (including beta=0).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    m0 = pid_m * BM
    n0 = pid_n * BN
    if UPLO == 0:
        if (m0 + (BM - 1)) < n0:
            return
    else:
        if m0 > (n0 + (BN - 1)):
            return

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_m = offs_m < N
    mask_n = offs_n < N
    mask_tile = mask_m[:, None] & mask_n[None, :]

    if UPLO == 0:
        mask_tri = offs_m[:, None] >= offs_n[None, :]
    else:
        mask_tri = offs_m[:, None] <= offs_n[None, :]
    mask = mask_tile & mask_tri

    k0 = pid_k * BK
    offs_k = k0 + tl.arange(0, BK)
    mask_k = offs_k < K

    # A_mT/B_mT: (BM,BK), A_n/B_n: (BK,BN)
    A_mT = tl.load(
        A_ptr + offs_k[None, :] * stride_a0 + offs_m[:, None] * stride_a1,
        mask=mask_m[:, None] & mask_k[None, :],
        other=0.0,
    ).to(tl.float64)
    B_mT = tl.load(
        B_ptr + offs_k[None, :] * stride_b0 + offs_m[:, None] * stride_b1,
        mask=mask_m[:, None] & mask_k[None, :],
        other=0.0,
    ).to(tl.float64)
    A_n = tl.load(
        A_ptr + offs_k[:, None] * stride_a0 + offs_n[None, :] * stride_a1,
        mask=mask_k[:, None] & mask_n[None, :],
        other=0.0,
    ).to(tl.float64)
    B_n = tl.load(
        B_ptr + offs_k[:, None] * stride_b0 + offs_n[None, :] * stride_b1,
        mask=mask_k[:, None] & mask_n[None, :],
        other=0.0,
    ).to(tl.float64)

    acc = tl.dot(A_mT, B_n).to(tl.float64) + tl.dot(B_mT, A_n).to(tl.float64)
    acc = acc * alpha

    # Atomic accumulate into C triangle.
    tl.atomic_add(
        C_ptr + offs_m[:, None] * stride_c0 + offs_n[None, :] * stride_c1,
        acc,
        mask=mask,
    )


def dsyr2k_triton_(
    C: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    uplo: str = "L",
    bm: int = 128,
    bn: int = 64,
    bk: int = 16,
    impl: str = "dot",  # "dot" | "dotk" | "outer"
    num_warps: int = 4,
    num_stages: int = 3,
):
    """
    Triton DSYR2K (trans='T') update on C (in-place).

    Shapes:
      C: (N,N) float64 CUDA
      A,B: (K,N) float64 CUDA (row-major contiguous recommended)

    Only the requested triangular part is updated.
    """
    assert C.is_cuda and A.is_cuda and B.is_cuda
    assert C.dtype == torch.float64 and A.dtype == torch.float64 and B.dtype == torch.float64
    assert C.ndim == 2 and C.shape[0] == C.shape[1]
    assert A.ndim == 2 and B.ndim == 2
    assert A.shape == B.shape
    K, N = A.shape
    assert C.shape[0] == N
    assert uplo in ("L", "U")
    uplo_flag = 0 if uplo == "L" else 1
    assert impl in ("dot", "dotk", "outer")

    grid_m = triton.cdiv(N, bm)
    grid_n = triton.cdiv(N, bn)
    grid = (grid_m, grid_n)
    kwargs = {}
    kwargs["num_warps"] = num_warps
    kwargs["num_stages"] = num_stages

    if impl == "outer":
        dsyr2k_tt_outer_kernel[grid](
            C, A, B,
            N,
            C.stride(0), C.stride(1),
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            alpha=alpha,
            beta=beta,
            UPLO=uplo_flag,
            K=K,
            BM=bm,
            BN=bn,
            **kwargs,
        )
    elif impl == "dot":
        dsyr2k_tt_dot_kernel[grid](
            C, A, B,
            N,
            C.stride(0), C.stride(1),
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            alpha=alpha,
            beta=beta,
            UPLO=uplo_flag,
            K=K,
            BK=bk,
            BM=bm,
            BN=bn,
            **kwargs,
        )
    else:
        # dotk uses atomics over the K dimension, so we need to apply beta first.
        # (This matches DSYR2K semantics on the updated triangle.)
        grid_scale = grid
        tri_scale_kernel[grid_scale](
            C,
            N,
            C.stride(0), C.stride(1),
            beta=beta,
            UPLO=uplo_flag,
            BM=bm,
            BN=bn,
            **kwargs,
        )

        grid_k = triton.cdiv(K, bk)
        grid3 = (grid_m, grid_n, grid_k)
        dsyr2k_tt_dotk_kernel[grid3](
            C, A, B,
            N,
            K,
            C.stride(0), C.stride(1),
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            alpha=alpha,
            UPLO=uplo_flag,
            BK=bk,
            BM=bm,
            BN=bn,
            **kwargs,
        )


def _bench_once(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) / iters


def _time_one_ms(fn):
    # 1 warm + 1 timed run, using CUDA events for accuracy.
    fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


TUNE_CONFIGS = [
    # (bm, bn, warps, stages)
    (128, 64, 4, 3),
    (128, 64, 8, 3),
    (64, 64, 4, 4),
    (256, 32, 4, 3),
    (64, 128, 4, 3),
]


def tune_dsyr2k(C, A, B, *, alpha, beta, uplo, bk=16, impl="dot", timeout_s=1.0):
    opt = None
    for (bm, bn, warps, stages) in TUNE_CONFIGS:
        def run():
            dsyr2k_triton_(
                C, A, B,
                alpha=alpha, beta=beta, uplo=uplo,
                bm=bm, bn=bn,
                bk=bk,
                impl=impl,
                num_warps=warps, num_stages=stages,
            )

        # First call pays JIT compilation cost; do not count it toward the per-config budget.
        run()
        torch.cuda.synchronize()

        # Now time just the steady-state kernel (warm once + run once).
        t_wall0 = time.time()
        t_ms = _time_one_ms(run)
        t_wall = time.time() - t_wall0
        # Timeout applies to runtime behavior, not compilation.
        if (t_ms / 1e3) > timeout_s:
            print(f"config bm={bm} bn={bn} warps={warps} stages={stages} "
                  f"TIMEOUT kernel={t_ms/1e3:.3f}s (skip)")
            continue

        print(f"config bm={bm} bn={bn} warps={warps} stages={stages} "
              f"time={t_ms:.3f} ms (wall {t_wall:.3f}s)")
        if opt is None or t_ms < opt[0]:
            opt = (t_ms, bm, bn, warps, stages)

    if opt is None:
        raise RuntimeError("No dsyr2k config finished within timeout.")
    return opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--uplo", type=str, default="L", choices=["L", "U"])
    ap.add_argument("--bm", type=int, default=128)
    ap.add_argument("--bn", type=int, default=64)
    ap.add_argument("--bk", type=int, default=16)
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--tune", action="store_true", help="run all configs: warm once + run once")
    ap.add_argument("--timeout-s", type=float, default=1.0, help="per-config wall-time budget during --tune")
    ap.add_argument("--impl", type=str, default="dot", choices=["dot", "dotk", "outer"])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(0)

    N, K = args.n, args.k
    C = torch.randn((N, N), device=device, dtype=torch.float64)
    # Make symmetric-ish to mimic typical use; only triangle is referenced by BLAS.
    C = 0.5 * (C + C.T)

    # A,B are (K,N) so that columns of the original (N,K) are contiguous.
    A = torch.randn((K, N), device=device, dtype=torch.float64)
    B = torch.randn((K, N), device=device, dtype=torch.float64)

    alpha = 1.0
    beta = 0.0

    if args.tune:
        opt = tune_dsyr2k(C, A, B, alpha=alpha, beta=beta, uplo=args.uplo, bk=args.bk, impl=args.impl, timeout_s=args.timeout_s)
        t_ms, bm, bn, warps, stages = opt
        flops = (N * (N + 1) // 2) * (4 * K)
        tflops = flops / (t_ms / 1e3) / 1e12
        ratio = tflops / 19.5
        print(f"opt: bm={bm} bn={bn} warps={warps} stages={stages} "
              f"time={t_ms:.3f} ms  approx {tflops:.2f} TFLOP/s  ({ratio*100:.1f}% of 19.5 TF)")
        return

    if args.check:
        C0 = C.clone()
        dsyr2k_triton_(C, A, B, alpha=alpha, beta=beta, uplo=args.uplo,
                       bm=args.bm, bn=args.bn,
                       bk=args.bk,
                       impl=args.impl,
                       num_warps=args.warps, num_stages=args.stages)
        # Reference via GEMMs, then only compare the updated triangle.
        Cref = beta * C0 + alpha * (A.T @ B + B.T @ A)
        if args.uplo == "L":
            mask = torch.tril(torch.ones((N, N), device=device, dtype=torch.bool))
        else:
            mask = torch.triu(torch.ones((N, N), device=device, dtype=torch.bool))
        max_abs = (C[mask] - Cref[mask]).abs().max().item()
        print(f"check max_abs={max_abs:.3e}")

    if args.bench:
        def run_triton():
            dsyr2k_triton_(C, A, B, alpha=alpha, beta=beta, uplo=args.uplo,
                           bm=args.bm, bn=args.bn,
                           bk=args.bk,
                           impl=args.impl,
                           num_warps=args.warps, num_stages=args.stages)

        def run_ref():
            # This writes full C (not just triangle) but gives a baseline.
            _ = beta * C + alpha * (A.T @ B + B.T @ A)

        # User preference: warm once + run once.
        t_triton = _bench_once(run_triton, warmup=1, iters=1)
        t_ref = _bench_once(run_ref, warmup=1, iters=1)

        # For uplo triangle only, flops ~ N*(N+1)/2 * 2*K (mul+add) * 2 terms?:
        # Each term is a dot of length K -> (2*K flops) per element; two terms -> ~4*K flops.
        # We update half matrix -> ~ N(N+1)/2 * 4K flops.
        flops = (N * (N + 1) // 2) * (4 * K)
        tflops = flops / t_triton / 1e12
        ratio = tflops / 19.5
        print(f"triton: {t_triton*1e3:.3f} ms  approx {tflops:.2f} TFLOP/s  ({ratio*100:.1f}% of 19.5 TF)")
        print(f"ref(gemm): {t_ref*1e3:.3f} ms")


if __name__ == "__main__":
    main()
