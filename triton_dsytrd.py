import argparse
import time

import torch
import triton
import triton.language as tl


@triton.jit
def dot_partial_kernel(x_ptr, y_ptr, out_ptr, n,
                       stride_x, stride_y,
                       BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs * stride_x, mask=mask, other=0.0).to(tl.float64)
    y = tl.load(y_ptr + offs * stride_y, mask=mask, other=0.0).to(tl.float64)
    acc = tl.sum(x * y, axis=0)
    tl.atomic_add(out_ptr, acc)


def triton_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """FP64 dot implemented with Triton (returns a 0-dim CUDA tensor)."""
    assert x.is_cuda and y.is_cuda
    assert x.dtype == torch.float64 and y.dtype == torch.float64
    assert x.ndim == 1 and y.ndim == 1 and x.numel() == y.numel()
    n = x.numel()
    out = torch.zeros((), device=x.device, dtype=torch.float64)
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    dot_partial_kernel[grid](
        x, y, out, n,
        x.stride(0), y.stride(0),
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out


@triton.jit
def axpby_kernel(x_ptr, y_ptr, out_ptr, n,
                 stride_x, stride_y, stride_o,
                 a, b,
                 BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs * stride_x, mask=mask, other=0.0).to(tl.float64)
    y = tl.load(y_ptr + offs * stride_y, mask=mask, other=0.0).to(tl.float64)
    aa = a.to(tl.float64)
    bb = b.to(tl.float64)
    tl.store(out_ptr + offs * stride_o, aa * x + bb * y, mask=mask)


def triton_axpby_(out: torch.Tensor, x: torch.Tensor, y: torch.Tensor, a: float, b: float):
    """out = a*x + b*y (FP64)."""
    assert out.is_cuda and x.is_cuda and y.is_cuda
    assert out.dtype == x.dtype == y.dtype == torch.float64
    assert out.ndim == x.ndim == y.ndim == 1
    assert out.numel() == x.numel() == y.numel()
    n = out.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    axpby_kernel[grid](
        x, y, out, n,
        x.stride(0), y.stride(0), out.stride(0),
        a=a, b=b,
        BLOCK=BLOCK,
        num_warps=4,
    )


@triton.jit
def matvec_partial_kernel(A_ptr, x_ptr, y_ptr,
                          M, N,
                          stride_am, stride_an,
                          stride_x, stride_y,
                          BLOCK_N: tl.constexpr):
    """y[m] += sum_{n in block} A[m,n] * x[n] via atomic adds (supports dynamic M/N)."""
    m = tl.program_id(0)
    pid_k = tl.program_id(1)
    if m >= M:
        return

    n0 = pid_k * BLOCK_N
    offs_n = n0 + tl.arange(0, BLOCK_N)
    mask = offs_n < N
    a = tl.load(A_ptr + m * stride_am + offs_n * stride_an, mask=mask, other=0.0)
    x = tl.load(x_ptr + offs_n * stride_x, mask=mask, other=0.0)
    acc = tl.sum(a * x, axis=0)
    tl.atomic_add(y_ptr + m * stride_y, acc)


@triton.jit
def rank2_update_kernel(A_ptr, v_ptr, w_ptr,
                        M, N,
                        stride_am, stride_an,
                        stride_v, stride_w,
                        BM: tl.constexpr, BN: tl.constexpr):
    """A -= v w^T + w v^T on an MxN tile (dense update)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an, mask=mask, other=0.0)
    v_m = tl.load(v_ptr + offs_m * stride_v, mask=mask_m, other=0.0)[:, None]
    w_m = tl.load(w_ptr + offs_m * stride_w, mask=mask_m, other=0.0)[:, None]
    v_n = tl.load(v_ptr + offs_n * stride_v, mask=mask_n, other=0.0)[None, :]
    w_n = tl.load(w_ptr + offs_n * stride_w, mask=mask_n, other=0.0)[None, :]

    a = a - v_m * w_n - w_m * v_n
    tl.store(A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an, a, mask=mask)


def _householder_vec(x: torch.Tensor):
    """
    Householder for real vector x (1D), returns (v, tau, beta) such that
    (I - tau v v^T) x = [beta, 0, ..., 0]^T, with v[0]=1.
    """
    assert x.ndim == 1
    dtype = x.dtype
    device = x.device

    alpha = x[0]
    if x.numel() == 1:
        v = torch.ones_like(x)
        tau = torch.zeros((), dtype=dtype, device=device)
        beta = alpha
        return v, tau, beta

    x_tail = x[1:]
    sigma = triton_dot(x_tail, x_tail)

    # Avoid any host sync (no .item(), no Python if on device values).
    # For sigma==0, the reflector is identity: tau=0, beta=alpha, v=[1,0,...].
    is_zero = (sigma == 0)

    norm = torch.sqrt(alpha * alpha + sigma)
    beta0 = torch.where(alpha <= 0, norm, -norm)
    beta = torch.where(is_zero, alpha, beta0)
    tau = torch.where(is_zero, torch.zeros((), dtype=dtype, device=device), (beta0 - alpha) / beta0)
    scale = torch.where(is_zero, torch.zeros((), dtype=dtype, device=device), 1.0 / (alpha - beta0))

    v = x.clone()
    v[0] = 1.0
    v[1:] = v[1:] * scale
    return v, tau, beta


def dsytrd_triton_(A: torch.Tensor, uplo: str = "L", *, block_n=1024, bm=64, bn=64):
    """
    In-place symmetric tridiagonal reduction (prototype).

    - A: (n,n) CUDA tensor, float64
    - uplo: 'L' (lower) or 'U' (upper); this prototype updates the full trailing submatrix.

    Returns (d, e) where T has diag d and offdiag e.
    """
    assert A.is_cuda and A.ndim == 2 and A.shape[0] == A.shape[1]
    assert A.dtype == torch.float64, "this prototype is float64-only"
    n = A.shape[0]
    assert uplo in ("L", "U")

    # d: diagonal, e: off-diagonal
    d = torch.empty((n,), device=A.device, dtype=A.dtype)
    e = torch.empty((n - 1,), device=A.device, dtype=A.dtype)

    # Save initial diagonal as we go; updated diagonal is in A after each step.
    for k in range(n - 1):
        if uplo == "L":
            x = A[k + 1 :, k]
        else:
            x = A[k, k + 1 :]

        v, tau, beta = _householder_vec(x)
        e[k] = beta
        d[k] = A[k, k]

        # Overwrite the relevant off-diagonal element (keep symmetry explicitly).
        if uplo == "L":
            A[k + 1, k] = beta
            A[k, k + 1] = beta
            if v.numel() > 1:
                A[k + 2 :, k] = v[1:]
        else:
            A[k, k + 1] = beta
            A[k + 1, k] = beta
            if v.numel() > 1:
                A[k, k + 2 :] = v[1:]

        # Nothing to update if tau == 0 or trailing size is 0.
        m = n - (k + 1)
        if m <= 1:
            continue
        # Avoid host sync; tau==0 is rare for random data, and the update becomes a no-op anyway.

        A22 = A[k + 1 :, k + 1 :]
        v_full = v  # length m

        # w = tau * A22 @ v
        w = torch.empty((m,), device=A.device, dtype=A.dtype)
        w.zero_()
        grid = (m, triton.cdiv(m, block_n))
        matvec_partial_kernel[grid](
            A22, v_full, w,
            M=m, N=m,
            stride_am=A22.stride(0), stride_an=A22.stride(1),
            stride_x=v_full.stride(0), stride_y=w.stride(0),
            BLOCK_N=block_n,
            num_warps=4,
        )
        w *= tau

        # w = w + alpha*v, alpha = -0.5 * tau * (v^T w)
        alpha = (-0.5) * tau * triton_dot(v_full, w)
        w += alpha * v_full

        # A22 -= v w^T + w v^T
        grid2 = (triton.cdiv(m, bm), triton.cdiv(m, bn))
        rank2_update_kernel[grid2](
            A22, v_full, w,
            M=m, N=m,
            stride_am=A22.stride(0), stride_an=A22.stride(1),
            stride_v=v_full.stride(0), stride_w=w.stride(0),
            BM=bm, BN=bn,
            num_warps=4,
        )

        # Optional: enforce symmetry numerically (cheap for debug, expensive for big n).
        # A22.copy_(0.5 * (A22 + A22.T))

    d[n - 1] = A[n - 1, n - 1]
    return d, e


def build_tridiag(d: torch.Tensor, e: torch.Tensor):
    n = d.numel()
    T = torch.zeros((n, n), device=d.device, dtype=d.dtype)
    T.diagonal().copy_(d)
    i = torch.arange(n - 1, device=d.device)
    T[i + 1, i] = e
    T[i, i + 1] = e
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--uplo", type=str, default="L", choices=["L", "U"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"

    A = torch.randn((args.n, args.n), device=device, dtype=torch.float64)
    A = 0.5 * (A + A.T)  # symmetric

    A0 = A.clone() if args.check else None

    if args.bench:
        torch.cuda.synchronize()
        t0 = time.time()

    d, e = dsytrd_triton_(A, uplo=args.uplo)

    if args.bench:
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"n={args.n} time={(t1 - t0)*1e3:.3f} ms")

    if args.check:
        # Similarity transform preserves eigenvalues; compare eigenvalues of A0 and tridiagonal T.
        T = build_tridiag(d, e)
        w0 = torch.linalg.eigvalsh(A0)
        w1 = torch.linalg.eigvalsh(T)
        max_abs = (w0 - w1).abs().max().item()
        rel = max_abs / w0.abs().max().item()
        print(f"eigval max_abs={max_abs:.3e} rel={rel:.3e}")


if __name__ == "__main__":
    main()
