import argparse
import time

import torch
import triton
import triton.language as tl

from triton_dsytrd import _householder_vec, matvec_partial_kernel, rank2_update_kernel, triton_axpby_, triton_dot
from triton_dsyr2k import dsyr2k_triton_


@triton.jit
def gemv_t_partial_kernel(A_ptr, v_ptr, y_ptr,
                          M, N,
                          stride_am, stride_an,
                          stride_v, stride_y,
                          BLOCK_M: tl.constexpr,
                          BLOCK_N: tl.constexpr):
    """
    y += A^T v, where A is (M,N) and v is (M).
    Each program handles a tile (BLOCK_M rows, BLOCK_N cols) and atomically accumulates into y.
    A is assumed to be row-major-ish (stride_an == 1) for optimal performance.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    v = tl.load(v_ptr + offs_m * stride_v, mask=mask_m, other=0.0).to(tl.float64)  # (BM,)
    a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an,
                mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float64)  # (BM,BN)
    # Reduce over rows to partial sums for each column.
    acc = tl.sum(a * v[:, None], axis=0)  # (BN,)
    tl.atomic_add(y_ptr + offs_n * stride_y, acc, mask=mask_n)


@triton.jit
def rank1_update_rect_kernel(A_ptr, v_ptr, y_ptr,
                             M, N,
                             stride_am, stride_an,
                             stride_v, stride_y,
                             tau,
                             BM: tl.constexpr, BN: tl.constexpr):
    """A -= tau * v y^T for A (M,N), v (M), y (N)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an, mask=mask, other=0.0).to(tl.float64)
    v = tl.load(v_ptr + offs_m * stride_v, mask=mask_m, other=0.0).to(tl.float64)[:, None]
    y = tl.load(y_ptr + offs_n * stride_y, mask=mask_n, other=0.0).to(tl.float64)[None, :]
    t = tau.to(tl.float64)
    a = a - t * v * y
    tl.store(A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an, a, mask=mask)


@triton.jit
def rank1_update_rect_tau_ptr_kernel(A_ptr, v_ptr, y_ptr, tau_ptr,
                                     M, N,
                                     stride_am, stride_an,
                                     stride_v, stride_y,
                                     BM: tl.constexpr, BN: tl.constexpr):
    """A -= tau * v y^T where tau is loaded from a 0-dim tensor pointer (avoids host sync)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an, mask=mask, other=0.0).to(tl.float64)
    v = tl.load(v_ptr + offs_m * stride_v, mask=mask_m, other=0.0).to(tl.float64)[:, None]
    y = tl.load(y_ptr + offs_n * stride_y, mask=mask_n, other=0.0).to(tl.float64)[None, :]
    t = tl.load(tau_ptr).to(tl.float64)
    a = a - t * v * y
    tl.store(A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an, a, mask=mask)


@triton.jit
def copy_transpose_kernel(src_ptr, dst_ptr,
                          M, N,
                          stride_sm, stride_sn,
                          stride_dm, stride_dn,
                          BM: tl.constexpr, BN: tl.constexpr):
    """dst (N,M) = src^T (N,M) for src (M,N)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    x = tl.load(src_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn, mask=mask, other=0.0)
    # Write transpose
    tl.store(dst_ptr + offs_n[:, None] * stride_dm + offs_m[None, :] * stride_dn, x.T, mask=mask.T)


@triton.jit
def givens_cs_kernel(B_ptr, cs_ptr,
                     i1, i2, k,
                     stride0, stride1):
    """Compute c,s to zero B[i2,k] using rows (i1,i2), store into cs_ptr[0:2]."""
    a = tl.load(B_ptr + i1 * stride0 + k * stride1).to(tl.float64)
    b = tl.load(B_ptr + i2 * stride0 + k * stride1).to(tl.float64)
    r = tl.sqrt(a * a + b * b)
    c = tl.where(r == 0.0, 1.0, a / r)
    s = tl.where(r == 0.0, 0.0, b / r)
    tl.store(cs_ptr + 0, c)
    tl.store(cs_ptr + 1, s)


@triton.jit
def apply_givens_rows_kernel(B_ptr, cs_ptr,
                             n,
                             i1, i2,
                             stride0, stride1,
                             BLOCK: tl.constexpr):
    """Apply G to rows (i1,i2): [row1;row2] = [[c s],[-s c]]*[row1;row2]."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    c = tl.load(cs_ptr + 0).to(tl.float64)
    s = tl.load(cs_ptr + 1).to(tl.float64)
    a = tl.load(B_ptr + i1 * stride0 + offs * stride1, mask=mask, other=0.0).to(tl.float64)
    b = tl.load(B_ptr + i2 * stride0 + offs * stride1, mask=mask, other=0.0).to(tl.float64)
    x = c * a + s * b
    y = -s * a + c * b
    tl.store(B_ptr + i1 * stride0 + offs * stride1, x, mask=mask)
    tl.store(B_ptr + i2 * stride0 + offs * stride1, y, mask=mask)


@triton.jit
def apply_givens_cols_kernel(B_ptr, cs_ptr,
                             n,
                             j1, j2,
                             stride0, stride1,
                             BLOCK: tl.constexpr):
    """Apply G to cols (j1,j2): [col1 col2] = [col1 col2]*[[c s],[-s c]]."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    c = tl.load(cs_ptr + 0).to(tl.float64)
    s = tl.load(cs_ptr + 1).to(tl.float64)
    a = tl.load(B_ptr + offs * stride0 + j1 * stride1, mask=mask, other=0.0).to(tl.float64)
    b = tl.load(B_ptr + offs * stride0 + j2 * stride1, mask=mask, other=0.0).to(tl.float64)
    x = c * a + s * b
    y = -s * a + c * b
    tl.store(B_ptr + offs * stride0 + j1 * stride1, x, mask=mask)
    tl.store(B_ptr + offs * stride0 + j2 * stride1, y, mask=mask)


@triton.jit
def apply_givens_rows_range_kernel(B_ptr, cs_ptr,
                                   start, count,
                                   i1, i2,
                                   stride0, stride1,
                                   BLOCK: tl.constexpr):
    """Apply G to rows (i1,i2) for columns in [start, start+count)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < count
    col = start + offs
    c = tl.load(cs_ptr + 0).to(tl.float64)
    s = tl.load(cs_ptr + 1).to(tl.float64)
    a = tl.load(B_ptr + i1 * stride0 + col * stride1, mask=mask, other=0.0).to(tl.float64)
    b = tl.load(B_ptr + i2 * stride0 + col * stride1, mask=mask, other=0.0).to(tl.float64)
    x = c * a + s * b
    y = -s * a + c * b
    tl.store(B_ptr + i1 * stride0 + col * stride1, x, mask=mask)
    tl.store(B_ptr + i2 * stride0 + col * stride1, y, mask=mask)


@triton.jit
def apply_givens_cols_range_kernel(B_ptr, cs_ptr,
                                   start, count,
                                   j1, j2,
                                   stride0, stride1,
                                   BLOCK: tl.constexpr):
    """Apply G to cols (j1,j2) for rows in [start, start+count)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < count
    row = start + offs
    c = tl.load(cs_ptr + 0).to(tl.float64)
    s = tl.load(cs_ptr + 1).to(tl.float64)
    a = tl.load(B_ptr + row * stride0 + j1 * stride1, mask=mask, other=0.0).to(tl.float64)
    b = tl.load(B_ptr + row * stride0 + j2 * stride1, mask=mask, other=0.0).to(tl.float64)
    x = c * a + s * b
    y = -s * a + c * b
    tl.store(B_ptr + row * stride0 + j1 * stride1, x, mask=mask)
    tl.store(B_ptr + row * stride0 + j2 * stride1, y, mask=mask)

@triton.jit
def set2_kernel(B_ptr,
                i, j,
                stride0, stride1,
                v: tl.constexpr):
    tl.store(B_ptr + i * stride0 + j * stride1, v)


@triton.jit
def symmetrize_upper_from_lower_kernel(A_ptr,
                                       N,
                                       stride0, stride1,
                                       off,
                                       BM: tl.constexpr, BN: tl.constexpr):
    """For the square submatrix A[off:off+N, off:off+N], set upper = lower^T (in-place)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = off + pid_m * BM + tl.arange(0, BM)
    cols = off + pid_n * BN + tl.arange(0, BN)
    mask_r = rows < (off + N)
    mask_c = cols < (off + N)
    mask = mask_r[:, None] & mask_c[None, :]

    # Only write strictly upper triangle.
    upper = rows[:, None] < cols[None, :]
    mask = mask & upper
    src = tl.load(A_ptr + cols[None, :] * stride0 + rows[:, None] * stride1, mask=mask, other=0.0).to(tl.float64)
    tl.store(A_ptr + rows[:, None] * stride0 + cols[None, :] * stride1, src, mask=mask)


def larft_forward_columnwise(V: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """
    Build T (upper triangular) for the compact WY representation (forward, columnwise):
      Q = I - V T V^T
    V: (m, b) with V[k,k]=1 and V[0:k, k]=0
    tau: (b,)
    """
    assert V.ndim == 2 and tau.ndim == 1
    m, b = V.shape
    assert tau.numel() == b
    T = torch.zeros((b, b), device=V.device, dtype=V.dtype)
    for i in range(b):
        T[i, i] = tau[i]
        if i == 0:
            continue
        v = V[i:, i]  # (m-i,)
        tmp = V[i:, :i].T @ v  # (i,)
        tmp = (-tau[i]) * tmp
        T[:i, i] = T[:i, :i] @ tmp
    return T


def larft_forward_columnwise_gram(V: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """
    Faster larft for our usage when b is small (<=64):
      - Compute Gram matrix G = V^T V once (BLAS3).
      - Use the standard recurrence on the small bxb matrices only.

    This matches larft_forward_columnwise(), but avoids b large GEMMs on tall skinny slices.
    """
    assert V.ndim == 2 and tau.ndim == 1
    m, b = V.shape
    k = min(m, b)
    if k == 0:
        return torch.empty((0, 0), device=V.device, dtype=V.dtype)
    tau = tau[:k]
    G = V[:, :k].T @ V[:, :k]  # (k,k)
    T = torch.zeros((k, k), device=V.device, dtype=V.dtype)
    T.diagonal().copy_(tau)
    for i in range(1, k):
        w = (-tau[i]) * G[:i, i]  # (i,)
        # T[:i, i] = T[:i, :i] @ w
        T[:i, i] = T[:i, :i] @ w
    return T


def larft_forward_columnwise_gram_inplace(V: torch.Tensor, tau: torch.Tensor,
                                         T_out: torch.Tensor, G_out: torch.Tensor):
    """
    In-place variant of larft_forward_columnwise_gram().
    Writes:
      - G_out[:k,:k] = V^T V
      - T_out[:k,:k] = T
    """
    assert V.ndim == 2 and tau.ndim == 1
    m, b = V.shape
    k = min(m, b)
    if k == 0:
        return 0
    tau = tau[:k]
    G = G_out[:k, :k]
    # G = V^T V
    torch.mm(V[:, :k].T, V[:, :k], out=G)

    T = T_out[:k, :k]
    T.zero_()
    T.diagonal().copy_(tau)
    for i in range(1, k):
        w = (-tau[i]) * G[:i, i]
        T[:i, i] = T[:i, :i] @ w
    return k


def panel_qr_wy(panel: torch.Tensor, *, larft: str = "gram",
                V_buf: torch.Tensor | None = None,
                W_buf: torch.Tensor | None = None,
                T_buf: torch.Tensor | None = None,
                G_buf: torch.Tensor | None = None):
    """
    QR factorization of (m,b) panel using torch.geqrf, returning V,W for WY:
      Q = I - V T V^T ; W = V T
    Returns (V, W, factors, tau) where 'factors' contains R (upper) and reflectors (lower).
    """
    factors, tau = torch.geqrf(panel)
    m, b = panel.shape
    k = min(m, b)
    if k == 0:
        return torch.empty((m, 0), device=panel.device, dtype=panel.dtype), torch.empty((m, 0), device=panel.device, dtype=panel.dtype), factors, tau

    # Build V from reflectors stored in 'factors'. Householder QR has implicit 1 on the diagonal.
    # Use preallocated workspace when provided to avoid per-panel allocations.
    if V_buf is None:
        V = torch.tril(factors, diagonal=-1)
        V[:k, :k] = V[:k, :k] + torch.eye(k, device=panel.device, dtype=panel.dtype)
        Vk = V[:, :k]
        if larft == "gram":
            T = larft_forward_columnwise_gram(Vk, tau[:k])
        elif larft == "loop":
            T = larft_forward_columnwise(Vk, tau[:k])
        else:
            raise ValueError(f"unknown larft={larft}")
        W = Vk @ T
        return Vk, W, factors, tau

    # Workspace path.
    assert W_buf is not None and T_buf is not None and G_buf is not None
    Vw = V_buf[:m, :b]
    Vw.copy_(factors)
    # Keep strictly lower part, then set diag to 1 for the k reflectors.
    Vw.tril_(diagonal=-1)
    Vw.diagonal()[:k].fill_(1.0)
    Vk = Vw[:, :k]

    if larft == "gram":
        larft_forward_columnwise_gram_inplace(Vk, tau[:k], T_buf, G_buf)
        T = T_buf[:k, :k]
    elif larft == "loop":
        T = larft_forward_columnwise(Vk, tau[:k])
    else:
        raise ValueError(f"unknown larft={larft}")

    W = W_buf[:m, :k]
    torch.mm(Vk, T, out=W)
    return Vk, W, factors, tau


def sbr_to_band_dbbr_triton_(A: torch.Tensor, b: int, *,
                             dsyr2k_cfg: dict | None = None,
                             symmetrize: bool = True,
                             update: str = "syr2k"):
    """
    Stage 1 (SBR/DBBR-style blocking): reduce symmetric A to symmetric band with lower bandwidth b (uplo=L),
    using panel QR (width b) and BLAS3-style trailing updates.

    Implementation notes:
      - Panel QR uses torch.geqrf (cuSOLVER) for now.
      - Trailing update uses Triton dsyr2k (fast path: impl='dot', BK=16).
      - To compute Z = A22 W - 0.5 V (W^T A22 W) we need A22 symmetric in storage; optionally
        mirror upper triangle from lower after each trailing update (cost is pure copy bandwidth).
    """
    assert A.is_cuda and A.ndim == 2 and A.shape[0] == A.shape[1]
    assert A.dtype == torch.float64
    n = A.shape[0]
    assert b >= 1 and b < n

    assert update in ("syr2k", "gemm2", "gemm1"), "update must be 'syr2k', 'gemm2', or 'gemm1'"
    cfg = dsyr2k_cfg or dict(bm=128, bn=64, bk=16, impl="dot", num_warps=4, num_stages=3)

    # Reuse workspaces to reduce allocator overhead.
    V_buf = torch.empty((n, b), device=A.device, dtype=A.dtype)
    W_buf = torch.empty((n, b), device=A.device, dtype=A.dtype)
    T_buf = torch.empty((b, b), device=A.device, dtype=A.dtype)
    G_buf = torch.empty((b, b), device=A.device, dtype=A.dtype)
    AW_buf = torch.empty((n, b), device=A.device, dtype=A.dtype)
    WT_buf = torch.empty((b, b), device=A.device, dtype=A.dtype)
    Z_buf = torch.empty((n, b), device=A.device, dtype=A.dtype)
    Vt_buf = torch.empty((b, n), device=A.device, dtype=A.dtype)  # (K,N) row-major
    Zt_buf = torch.empty((b, n), device=A.device, dtype=A.dtype)
    X_buf = torch.empty((n, 2 * b), device=A.device, dtype=A.dtype)
    Y_buf = torch.empty((n, 2 * b), device=A.device, dtype=A.dtype)

    # Process block columns i:i+b, where the "panel" sits b rows below the diagonal.
    for i in range(0, n - b - 1, b):
        base = i + b
        if base >= n:
            break
        m = n - base
        if m <= 0:
            break

        # Red panel: A(base:n, i:i+b)
        cols1 = min(i + b, n)
        panel = A[base:, i:cols1]
        V, W, factors, tau = panel_qr_wy(panel, V_buf=V_buf[base:, :], W_buf=W_buf[base:, :], T_buf=T_buf, G_buf=G_buf)
        panel.copy_(factors)

        # Two-sided update on trailing block A22 (size m x m):
        A22 = A[base:, base:]

        # Compute Z for the symmetric update:
        #   Z = A22 @ W - 0.5 * V @ (W^T @ (A22 @ W))
        # Shapes: V,W,Z are (m,k), k<=b.
        k = V.shape[1]
        if k == 0:
            continue

        # AW = A22 @ W
        AW = AW_buf[base:, :k]
        torch.mm(A22, W, out=AW)
        WT_AW = WT_buf[:k, :k]
        torch.mm(W.T, AW, out=WT_AW)  # (k,k)
        Z = Z_buf[base:, :k]
        Z.copy_(AW)
        Z.addmm_(V, WT_AW, beta=1.0, alpha=-0.5)

        if update == "syr2k":
            # A22 -= V Z^T + Z V^T, symmetric rank-2k update.
            # dsyr2k wants A,B as (K,N): (k,m)
            m = A22.shape[0]
            VkN = Vt_buf[:k, :m]
            ZkN = Zt_buf[:k, :m]
            VkN.copy_(V.T)
            ZkN.copy_(Z.T)
            dsyr2k_triton_(
                A22,
                VkN, ZkN,
                alpha=-1.0, beta=1.0, uplo="L",
                **cfg,
            )

            if symmetrize:
                # Keep A22 symmetric in storage for subsequent GEMMs (AW computation).
                grid = (triton.cdiv(m, 128), triton.cdiv(m, 128))
                symmetrize_upper_from_lower_kernel[grid](
                    A,  # full matrix pointer
                    m,
                    A.stride(0), A.stride(1),
                    off=base,
                    BM=128, BN=128,
                    num_warps=4,
                )
        elif update == "gemm2":
            # Dense update using 2 GEMMs:
            #   A22 := A22 - V Z^T - Z V^T
            # This keeps A22 symmetric in storage, and tends to hit a better GEMM fast path on GPUs.
            # Note: costs ~2x the flops vs SYR2K (updates full matrix), but may still be faster.
            A22.addmm_(V, Z.T, beta=1.0, alpha=-1.0)
            A22.addmm_(Z, V.T, beta=1.0, alpha=-1.0)
        else:
            # Same dense update, but expressed as a single GEMM with K=2b:
            #   A22 := A22 - [V Z] [Z V]^T
            # This has the same flop count as 2 GEMMs, but often maps to a better GEMM kernel.
            X = X_buf[base:, : 2 * k]
            Y = Y_buf[base:, : 2 * k]
            X[:, :k] = V
            X[:, k: 2 * k] = Z
            Y[:, :k] = Z
            Y[:, k: 2 * k] = V
            A22.addmm_(X, Y.T, beta=1.0, alpha=-1.0)




# ----------------------------
# Band-storage bulge chasing (MAGMA-like)
# ----------------------------

@triton.jit
def band_rect_matvec_kernel(A_ptr, x_ptr, y_ptr,
                            M, N,
                            stride0, stride1,  # band storage strides for (diff, col)
                            row0, col0,
                            BLOCK_N: tl.constexpr):
    """
    y[m] = sum_n A(row0+m, col0+n) * x[n]
    where A is stored in lower band layout (diff, col): A(m,n) at diff=(m-n), col=n.

    For this rectangular region, entries are contiguous with:
      base = A(row0, col0), row stride = stride0, col stride = (stride1 - stride0).
    """
    m = tl.program_id(0)
    if m >= M:
        return
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N

    # A(row0+m, col0+offs_n) => diff = (row0+m) - (col0+offs_n)
    diff = (row0 + m) - (col0 + offs_n)
    a = tl.load(A_ptr + diff * stride0 + (col0 + offs_n) * stride1, mask=mask, other=0.0).to(tl.float64)
    x = tl.load(x_ptr + offs_n, mask=mask, other=0.0).to(tl.float64)
    acc = tl.sum(a * x, axis=0)
    tl.store(y_ptr + m, acc)


@triton.jit
def band_rect_rmatvec_kernel(A_ptr, x_ptr, y_ptr,
                             M, N,
                             stride0, stride1,
                             row0, col0,
                             BLOCK_M: tl.constexpr):
    """
    y[n] = sum_m A(row0+m, col0+n) * x[m]  == (A^T x)[n]
    """
    n = tl.program_id(0)
    if n >= N:
        return
    offs_m = tl.arange(0, BLOCK_M)
    mask = offs_m < M
    diff = (row0 + offs_m) - (col0 + n)
    a = tl.load(A_ptr + diff * stride0 + (col0 + n) * stride1, mask=mask, other=0.0).to(tl.float64)
    x = tl.load(x_ptr + offs_m, mask=mask, other=0.0).to(tl.float64)
    acc = tl.sum(a * x, axis=0)
    tl.store(y_ptr + n, acc)


@triton.jit
def band_rect_rank1_update_kernel(A_ptr, u_ptr, y_ptr,
                                  M, N,
                                  stride0, stride1,
                                  row0, col0,
                                  tau,
                                  BM: tl.constexpr, BN: tl.constexpr):
    """A(row0:row0+M, col0:col0+N) -= tau * u y^T (rank-1 update) in band storage region."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    u = tl.load(u_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float64)[:, None]
    y = tl.load(y_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float64)[None, :]
    t = tau.to(tl.float64)

    # Load A at the mapped (row,col) positions.
    diff = (row0 + offs_m[:, None]) - (col0 + offs_n[None, :])
    a = tl.load(A_ptr + diff * stride0 + (col0 + offs_n[None, :]) * stride1, mask=mask, other=0.0).to(tl.float64)
    a = a - t * u * y
    tl.store(A_ptr + diff * stride0 + (col0 + offs_n[None, :]) * stride1, a, mask=mask)


@triton.jit
def band_rect_rank1_update_tau_idx_kernel(A_ptr, u_ptr, y_ptr, tau_ptr, tau_idx,
                                          M, N,
                                          stride0, stride1,
                                          row0, col0,
                                          BM: tl.constexpr, BN: tl.constexpr):
    """Band rank-1 update with tau loaded from tau_ptr[tau_idx] (avoids host sync)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    u = tl.load(u_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float64)[:, None]
    y = tl.load(y_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float64)[None, :]
    t = tl.load(tau_ptr + tau_idx).to(tl.float64)

    diff = (row0 + offs_m[:, None]) - (col0 + offs_n[None, :])
    a = tl.load(A_ptr + diff * stride0 + (col0 + offs_n[None, :]) * stride1, mask=mask, other=0.0).to(tl.float64)
    a = a - t * u * y
    tl.store(A_ptr + diff * stride0 + (col0 + offs_n[None, :]) * stride1, a, mask=mask)


@triton.jit
def band_larfx_left_fused_kernel(A_ptr, v_ptr, tau_ptr,
                                 M, N,
                                 stride0, stride1,
                                 row0, col0,
                                 BM: tl.constexpr, BN: tl.constexpr):
    """
    Fused left Householder on a band-stored rectangular region with guaranteed row>col:
      C := (I - tau v v^T) C, where C is A[row0:row0+M, col0:col0+N].
    Computes y = v^T C, then C -= tau * v y.
    """
    offs_m = tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    v = tl.load(v_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float64)  # (M,)
    t = tl.load(tau_ptr).to(tl.float64)

    diff = (row0 + offs_m[:, None]) - (col0 + offs_n[None, :])
    a = tl.load(A_ptr + diff * stride0 + (col0 + offs_n[None, :]) * stride1, mask=mask, other=0.0).to(tl.float64)

    # y[n] = sum_m v[m] * a[m,n]
    y = tl.sum(a * v[:, None], axis=0)  # (BN,)
    a = a - t * (v[:, None] * y[None, :])
    tl.store(A_ptr + diff * stride0 + (col0 + offs_n[None, :]) * stride1, a, mask=mask)


@triton.jit
def band_larfx_right_fused_kernel(A_ptr, v_ptr, tau_ptr,
                                  M, N,
                                  stride0, stride1,
                                  row0, col0,
                                  BM: tl.constexpr, BN: tl.constexpr):
    """
    Fused right Householder on a band-stored rectangular region with guaranteed row>col:
      C := C (I - tau v v^T), where C is A[row0:row0+M, col0:col0+N].
    Computes y = C v, then C -= tau * y v^T.
    """
    offs_m = tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    v = tl.load(v_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float64)  # (N,)
    t = tl.load(tau_ptr).to(tl.float64)

    diff = (row0 + offs_m[:, None]) - (col0 + offs_n[None, :])
    a = tl.load(A_ptr + diff * stride0 + (col0 + offs_n[None, :]) * stride1, mask=mask, other=0.0).to(tl.float64)

    # y[m] = sum_n a[m,n] * v[n]
    y = tl.sum(a * v[None, :], axis=1)  # (BM,)
    a = a - t * (y[:, None] * v[None, :])
    tl.store(A_ptr + diff * stride0 + (col0 + offs_n[None, :]) * stride1, a, mask=mask)


@triton.jit
def band_larfy_lower_fused_kernel(A_ptr, v_ptr, tau_ptr,
                                  N,
                                  stride0, stride1,
                                  col0,
                                  BLK: tl.constexpr):
    """
    Fused symmetric Householder update on the diagonal block A(col0:col0+N, col0:col0+N),
    stored in lower band format:
      A := (I - tau v v^T) A (I - tau v v^T)
    Only updates the stored lower triangle.
    """
    i = tl.arange(0, BLK)
    j = tl.arange(0, BLK)
    mask_i = i < N
    mask_j = j < N
    mask = mask_i[:, None] & mask_j[None, :]

    # Reconstruct dense symmetric block (BLK,BLK) in registers.
    ii = tl.maximum(i[:, None], j[None, :])
    jj = tl.minimum(i[:, None], j[None, :])
    diff = (col0 + ii) - (col0 + jj)
    Aij = tl.load(A_ptr + diff * stride0 + (col0 + jj) * stride1, mask=mask, other=0.0).to(tl.float64)

    v = tl.load(v_ptr + j, mask=mask_j, other=0.0).to(tl.float64)  # (BLK,)
    t = tl.load(tau_ptr).to(tl.float64)

    # w = tau * A * v
    w = tl.sum(Aij * v[None, :], axis=1)  # (BLK,)
    w = w * t

    # alpha = -0.5 * tau * (v^T w)
    vt_w = tl.sum(v * w, axis=0)
    alpha = (-0.5) * t * vt_w
    w = w + alpha * v

    # A -= v w^T + w v^T (only lower triangle written back)
    Anew = Aij - v[None, :] * w[:, None] - w[None, :] * v[:, None]
    lower = (i[:, None] >= j[None, :])
    mask_store = mask & lower
    tl.store(A_ptr + diff * stride0 + (col0 + jj) * stride1, Anew, mask=mask_store)


@triton.jit
def hbtype1_real_fused_body(A_ptr, V_ptr, TAU_ptr, W_ptr,
                            n, nb,
                            st, ed, sweep,
                            stride0, stride1,
                            MOD_STORE: tl.constexpr,
                            BLK: tl.constexpr):
    """
    Fused real hbtype1cb (wantz=0) for MAGMA band storage.
    - Builds Householder v/tau from column (st-1) over rows st:ed.
    - Stores v in V, tau in TAU.
    - Writes beta back to A(st,st-1) and zeros below.
    - Applies symmetric update on the diagonal block A(st:ed, st:ed) (stored lower).
    Indices are 0-based.
    """
    # All work is small (<=64); one program is enough.
    col = st - 1
    if col < 0:
        return
    len_ = ed - st + 1
    if len_ <= 1:
        return

    offs = tl.arange(0, BLK)
    mask_len = offs < len_

    slot = sweep % MOD_STORE
    vpos = slot * n + st
    taupos = slot * n + st

    # x[k] corresponds to A(st+k, col), stored at diff=(k+1) for column 'col'.
    x = tl.load(A_ptr + (offs + 1) * stride0 + col * stride1, mask=mask_len, other=0.0).to(tl.float64)
    # Triton doesn't currently support scalar indexing like x[0] in all backends.
    alpha = tl.load(A_ptr + 1 * stride0 + col * stride1).to(tl.float64)
    xt = tl.where(offs > 0, x, 0.0)
    sigma = tl.sum(xt * xt, axis=0)
    is_zero = sigma == 0.0

    norm = tl.sqrt(alpha * alpha + sigma)
    beta0 = tl.where(alpha <= 0.0, norm, -norm)
    beta = tl.where(is_zero, alpha, beta0)
    tau0 = (beta0 - alpha) / beta0
    tau = tl.where(is_zero, 0.0, tau0)
    scale = tl.where(is_zero, 0.0, 1.0 / (alpha - beta0))

    v = tl.where(offs == 0, 1.0, x * scale).to(tl.float64)

    # Store v and tau.
    tl.store(V_ptr + vpos + offs, v, mask=mask_len)
    tl.store(TAU_ptr + taupos, tau)

    # Write beta to A(st, col) => diff=1, and zero below it (diff>=2).
    tl.store(A_ptr + 1 * stride0 + col * stride1, beta)
    mask_zero = (offs >= 1) & mask_len
    tl.store(A_ptr + (offs + 1) * stride0 + col * stride1, 0.0, mask=mask_zero)

    # Apply symmetric update on diagonal block A(st:ed, st:ed) stored in band format.
    #
    # IMPORTANT: avoid materializing a full 64x64 block in registers (can spill badly in fp64).
    # Do the symmetric matvec + rank-2 update in tiles.
    BM: tl.constexpr = 16
    BN: tl.constexpr = 16
    BK: tl.constexpr = 16

    # w = tau * A * v (A is symmetric; only lower stored, so load via symmetry)
    w = tl.zeros([BLK], dtype=tl.float64)
    r = tl.arange(0, BLK)
    mask_r = r < len_
    for k0 in tl.static_range(0, BLK, BK):
        c = k0 + tl.arange(0, BK)
        mask_c = c < len_
        rr = r[:, None]
        cc = c[None, :]
        # Load A(st+rr, st+cc) using symmetry (ii=max, jj=min).
        ii = tl.maximum(rr, cc)
        jj = tl.minimum(rr, cc)
        diff = (ii - jj).to(tl.int32)
        Ablk = tl.load(A_ptr + diff * stride0 + (st + jj) * stride1,
                       mask=mask_r[:, None] & mask_c[None, :], other=0.0).to(tl.float64)
        vblk = tl.load(V_ptr + vpos + c, mask=mask_c, other=0.0).to(tl.float64)
        w += tl.sum(Ablk * vblk[None, :], axis=1)
    w = w * tau

    vt_w = tl.sum(v * w, axis=0)
    alpha2 = (-0.5) * tau * vt_w
    w = w + alpha2 * v

    # Store w so we can reload tiles without relying on unsupported register indexing.
    wpos = slot * n + st
    tl.store(W_ptr + wpos + offs, w, mask=mask_len)

    # A -= v w^T + w v^T (store only lower triangle).
    for i0 in tl.static_range(0, BLK, BM):
        ii = i0 + tl.arange(0, BM)
        mask_i = ii < len_
        v_i = tl.load(V_ptr + vpos + ii, mask=mask_i, other=0.0).to(tl.float64)
        w_i = tl.load(W_ptr + wpos + ii, mask=mask_i, other=0.0).to(tl.float64)
        for j0 in tl.static_range(0, BLK, BN):
            jj = j0 + tl.arange(0, BN)
            mask_j = jj < len_
            v_j = tl.load(V_ptr + vpos + jj, mask=mask_j, other=0.0).to(tl.float64)
            w_j = tl.load(W_ptr + wpos + jj, mask=mask_j, other=0.0).to(tl.float64)

            # For the diagonal block, (st+ii, st+jj) stays within the stored lower band.
            # Only write lower triangle (ii >= jj).
            ii2 = ii[:, None]
            jj2 = jj[None, :]
            mask = mask_i[:, None] & mask_j[None, :] & (ii2 >= jj2)
            diff2 = (ii2 - jj2).to(tl.int32)
            Ablk = tl.load(A_ptr + diff2 * stride0 + (st + jj2) * stride1, mask=mask, other=0.0).to(tl.float64)
            Ablk = Ablk - v_i[:, None] * w_j[None, :] - w_i[:, None] * v_j[None, :]
            tl.store(A_ptr + diff2 * stride0 + (st + jj2) * stride1, Ablk, mask=mask)

@triton.jit
def hbtype1_real_fused_kernel(A_ptr, V_ptr, TAU_ptr, W_ptr,
                              n, nb,
                              st, ed, sweep,
                              stride0, stride1,
                              MOD_STORE: tl.constexpr,
                              BLK: tl.constexpr):
    hbtype1_real_fused_body(
        A_ptr, V_ptr, TAU_ptr, W_ptr,
        n, nb,
        st, ed, sweep,
        stride0, stride1,
        MOD_STORE=MOD_STORE, BLK=BLK,
    )


@triton.jit
def hbtype3_real_fused_body(A_ptr, V_ptr, TAU_ptr, W_ptr,
                            n, nb,
                            st, ed, sweep,
                            stride0, stride1,
                            MOD_STORE: tl.constexpr,
                            BLK: tl.constexpr):
    """Fused real hbtype3cb: apply symmetric update on diagonal block using stored v/tau."""
    len_ = ed - st + 1
    if len_ <= 1:
        return
    offs = tl.arange(0, BLK)
    mask_len = offs < len_

    slot = sweep % MOD_STORE
    vpos = slot * n + st
    taupos = slot * n + st
    v = tl.load(V_ptr + vpos + offs, mask=mask_len, other=0.0).to(tl.float64)
    tau = tl.load(TAU_ptr + taupos).to(tl.float64)

    # Tiled symmetric update to avoid full 64x64 register materialization (see hbtype1).
    BM: tl.constexpr = 16
    BN: tl.constexpr = 16
    BK: tl.constexpr = 16

    w = tl.zeros([BLK], dtype=tl.float64)
    r = tl.arange(0, BLK)
    mask_r = r < len_
    for k0 in tl.static_range(0, BLK, BK):
        c = k0 + tl.arange(0, BK)
        mask_c = c < len_
        rr = r[:, None]
        cc = c[None, :]
        ii = tl.maximum(rr, cc)
        jj = tl.minimum(rr, cc)
        diff = (ii - jj).to(tl.int32)
        Ablk = tl.load(A_ptr + diff * stride0 + (st + jj) * stride1,
                       mask=mask_r[:, None] & mask_c[None, :], other=0.0).to(tl.float64)
        # Reload v tile from memory (avoid register indexing).
        vblk = tl.load(V_ptr + vpos + c, mask=mask_c, other=0.0).to(tl.float64)
        w += tl.sum(Ablk * vblk[None, :], axis=1)
    w = w * tau

    vt_w = tl.sum(v * w, axis=0)
    alpha2 = (-0.5) * tau * vt_w
    w = w + alpha2 * v

    wpos = slot * n + st
    tl.store(W_ptr + wpos + offs, w, mask=mask_len)

    for i0 in tl.static_range(0, BLK, BM):
        ii = i0 + tl.arange(0, BM)
        mask_i = ii < len_
        v_i = tl.load(V_ptr + vpos + ii, mask=mask_i, other=0.0).to(tl.float64)
        w_i = tl.load(W_ptr + wpos + ii, mask=mask_i, other=0.0).to(tl.float64)
        for j0 in tl.static_range(0, BLK, BN):
            jj = j0 + tl.arange(0, BN)
            mask_j = jj < len_
            v_j = tl.load(V_ptr + vpos + jj, mask=mask_j, other=0.0).to(tl.float64)
            w_j = tl.load(W_ptr + wpos + jj, mask=mask_j, other=0.0).to(tl.float64)

            ii2 = ii[:, None]
            jj2 = jj[None, :]
            mask = mask_i[:, None] & mask_j[None, :] & (ii2 >= jj2)
            diff2 = (ii2 - jj2).to(tl.int32)
            Ablk = tl.load(A_ptr + diff2 * stride0 + (st + jj2) * stride1, mask=mask, other=0.0).to(tl.float64)
            Ablk = Ablk - v_i[:, None] * w_j[None, :] - w_i[:, None] * v_j[None, :]
            tl.store(A_ptr + diff2 * stride0 + (st + jj2) * stride1, Ablk, mask=mask)

@triton.jit
def hbtype3_real_fused_kernel(A_ptr, V_ptr, TAU_ptr, W_ptr,
                              n, nb,
                              st, ed, sweep,
                              stride0, stride1,
                              MOD_STORE: tl.constexpr,
                              BLK: tl.constexpr):
    hbtype3_real_fused_body(
        A_ptr, V_ptr, TAU_ptr, W_ptr,
        n, nb,
        st, ed, sweep,
        stride0, stride1,
        MOD_STORE=MOD_STORE, BLK=BLK,
    )


@triton.jit
def hbtype2_real_fused_body(A_ptr, V_ptr, TAU_ptr,
                            n, nb,
                            st, ed, sweep,
                            stride0, stride1,
                            MOD_STORE: tl.constexpr,
                            BLK: tl.constexpr):
    """
    Fused real hbtype2cb (wantz=0) for MAGMA band storage.
    Applies:
      - right update on A(J1:J2, st:ed) with stored v/tau at (st)
      - generate bulge reflector at column st and store it at (J1)
      - left update on A(J1:J2, st+1:ed) with new reflector
    """
    len_ = ed - st + 1
    if len_ <= 0:
        return
    J1 = ed + 1
    if J1 >= n:
        return
    J2 = tl.minimum(ed + nb, n - 1)
    lem = J2 - J1 + 1
    if lem <= 0:
        return

    slot = sweep % MOD_STORE
    vpos = slot * n + st
    taupos = slot * n + st

    BK: tl.constexpr = 16
    BM: tl.constexpr = 16
    BN: tl.constexpr = 16

    # Load tau (scalar) from the top block (reflector at st).
    tautop = tl.load(TAU_ptr + taupos).to(tl.float64)

    # Right update on C = A(J1:J2, st:ed), shape (lem, len_):
    #   y = C * vtop
    #   C -= tautop * y * vtop^T
    #
    # Do it in tiles to avoid holding the full (<=64x64) C in registers.
    r = tl.arange(0, BLK)
    mask_r = r < lem
    y = tl.zeros([BLK], dtype=tl.float64)
    for k0 in tl.static_range(0, BLK, BK):
        c = k0 + tl.arange(0, BK)
        mask_c = c < len_
        row_abs = (J1 + r).to(tl.int32)[:, None]
        col_abs = (st + c).to(tl.int32)[None, :]
        diff = (row_abs - col_abs).to(tl.int32)
        Cblk = tl.load(A_ptr + diff * stride0 + col_abs * stride1,
                       mask=mask_r[:, None] & mask_c[None, :], other=0.0).to(tl.float64)
        vblk = tl.load(V_ptr + vpos + c, mask=mask_c, other=0.0).to(tl.float64)
        y += tl.sum(Cblk * vblk[None, :], axis=1)

    y = tl.where(mask_r, y, 0.0).to(tl.float64)
    for k0 in tl.static_range(0, BLK, BK):
        c = k0 + tl.arange(0, BK)
        mask_c = c < len_
        row_abs = (J1 + r).to(tl.int32)[:, None]
        col_abs = (st + c).to(tl.int32)[None, :]
        diff = (row_abs - col_abs).to(tl.int32)
        Cblk = tl.load(A_ptr + diff * stride0 + col_abs * stride1,
                       mask=mask_r[:, None] & mask_c[None, :], other=0.0).to(tl.float64)
        vblk = tl.load(V_ptr + vpos + c, mask=mask_c, other=0.0).to(tl.float64)
        Cblk = Cblk - tautop * (y[:, None] * vblk[None, :])
        tl.store(A_ptr + diff * stride0 + col_abs * stride1, Cblk,
                 mask=mask_r[:, None] & mask_c[None, :])

    if lem <= 1:
        return

    # Generate reflector from bulge column at col=st, rows J1:J2 (length lem).
    offs_l = tl.arange(0, BLK)
    mask_l = offs_l < lem
    diff0 = (J1 - st) + offs_l  # (row-col) for entries A(J1+offs, st)
    x = tl.load(A_ptr + diff0 * stride0 + st * stride1, mask=mask_l, other=0.0).to(tl.float64)
    alpha = tl.load(A_ptr + (J1 - st) * stride0 + st * stride1).to(tl.float64)
    xt = tl.where(offs_l > 0, x, 0.0)
    sigma = tl.sum(xt * xt, axis=0)
    is_zero = sigma == 0.0

    norm = tl.sqrt(alpha * alpha + sigma)
    beta0 = tl.where(alpha <= 0.0, norm, -norm)
    beta = tl.where(is_zero, alpha, beta0)
    tau0 = (beta0 - alpha) / beta0
    tau = tl.where(is_zero, 0.0, tau0)
    scale = tl.where(is_zero, 0.0, 1.0 / (alpha - beta0))
    v = tl.where(offs_l == 0, 1.0, x * scale).to(tl.float64)

    vpos2 = slot * n + J1
    taupos2 = slot * n + J1
    tl.store(V_ptr + vpos2 + offs_l, v, mask=mask_l)
    tl.store(TAU_ptr + taupos2, tau)

    # Write beta to A(J1,st) and zero below it in that column (J1+1..J2).
    tl.store(A_ptr + (J1 - st) * stride0 + st * stride1, beta)
    mask_zero = (offs_l >= 1) & mask_l
    tl.store(A_ptr + diff0 * stride0 + st * stride1, 0.0, mask=mask_zero)

    # Left update on A(J1:J2, st+1:ed), shape (lem, len_-1):
    #   y2 = v^T * L
    #   L -= tau * v * y2
    len2 = len_ - 1
    if len2 <= 0:
        return

    # Process columns in tiles so we don't materialize the full L.
    for j0 in tl.static_range(0, BLK, BN):
        jj = j0 + tl.arange(0, BN)
        mask_j = jj < len2
        y2 = tl.zeros([BN], dtype=tl.float64)

        # y2[j] = sum_i v[i] * L[i,j]
        for i0 in tl.static_range(0, BLK, BM):
            ii = i0 + tl.arange(0, BM)
            mask_i = ii < lem
            row_abs = (J1 + ii).to(tl.int32)[:, None]
            col_abs = ((st + 1) + jj).to(tl.int32)[None, :]
            diff = (row_abs - col_abs).to(tl.int32)
            Lblk = tl.load(A_ptr + diff * stride0 + col_abs * stride1,
                           mask=mask_i[:, None] & mask_j[None, :], other=0.0).to(tl.float64)
            v_i = tl.load(V_ptr + vpos2 + ii, mask=mask_i, other=0.0).to(tl.float64)
            y2 += tl.sum(Lblk * v_i[:, None], axis=0)

        # Update L tiles: L -= tau * v * y2
        for i0 in tl.static_range(0, BLK, BM):
            ii = i0 + tl.arange(0, BM)
            mask_i = ii < lem
            row_abs = (J1 + ii).to(tl.int32)[:, None]
            col_abs = ((st + 1) + jj).to(tl.int32)[None, :]
            diff = (row_abs - col_abs).to(tl.int32)
            Lblk = tl.load(A_ptr + diff * stride0 + col_abs * stride1,
                           mask=mask_i[:, None] & mask_j[None, :], other=0.0).to(tl.float64)
            v_i = tl.load(V_ptr + vpos2 + ii, mask=mask_i, other=0.0).to(tl.float64)
            Lblk = Lblk - tau * (v_i[:, None] * y2[None, :])
            tl.store(A_ptr + diff * stride0 + col_abs * stride1, Lblk,
                     mask=mask_i[:, None] & mask_j[None, :])

@triton.jit
def hbtype2_real_fused_kernel(A_ptr, V_ptr, TAU_ptr,
                              n, nb,
                              st, ed, sweep,
                              stride0, stride1,
                              MOD_STORE: tl.constexpr,
                              BLK: tl.constexpr):
    hbtype2_real_fused_body(
        A_ptr, V_ptr, TAU_ptr,
        n, nb,
        st, ed, sweep,
        stride0, stride1,
        MOD_STORE=MOD_STORE, BLK=BLK,
    )


@triton.jit
def band_symv_lower_kernel(A_ptr, v_ptr, y_ptr,
                           N,
                           stride0, stride1,
                           col0,
                           BLOCK: tl.constexpr):
    """
    y[r] = sum_c A(col0+r, col0+c) * v[c] for symmetric A stored in lower band layout.
    Only reads stored lower triangle via symmetry: A(i,j) = A(j,i).
    """
    r = tl.program_id(0)
    if r >= N:
        return
    c = tl.arange(0, BLOCK)
    mask = c < N

    rr = tl.full([BLOCK], r, tl.int32)
    i = tl.maximum(rr, c)
    j = tl.minimum(rr, c)
    diff = (col0 + i) - (col0 + j)  # i-j
    a = tl.load(A_ptr + diff * stride0 + (col0 + j) * stride1, mask=mask, other=0.0).to(tl.float64)
    v = tl.load(v_ptr + c, mask=mask, other=0.0).to(tl.float64)
    acc = tl.sum(a * v, axis=0)
    tl.store(y_ptr + r, acc)


@triton.jit
def band_sym_rank2_lower_update_kernel(A_ptr, v_ptr, w_ptr,
                                       N,
                                       stride0, stride1,
                                       col0,
                                       BM: tl.constexpr, BN: tl.constexpr):
    """A(col0:col0+N, col0:col0+N) -= v w^T + w v^T on the stored lower triangle."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask_m = offs_m < N
    mask_n = offs_n < N

    mm = offs_m[:, None]
    nn = offs_n[None, :]
    mask = mask_m[:, None] & mask_n[None, :] & (mm >= nn)

    v_m = tl.load(v_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float64)[:, None]
    w_m = tl.load(w_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float64)[:, None]
    v_n = tl.load(v_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float64)[None, :]
    w_n = tl.load(w_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float64)[None, :]

    diff = (mm - nn).to(tl.int32)
    a = tl.load(A_ptr + diff * stride0 + (col0 + nn) * stride1, mask=mask, other=0.0).to(tl.float64)
    a = a - v_m * w_n - w_m * v_n
    tl.store(A_ptr + diff * stride0 + (col0 + nn) * stride1, a, mask=mask)


def _zero_below_band_(A: torch.Tensor, i: int, b: int):
    # For column i, ensure entries below i+b are zero (and symmetric upper entries).
    n = A.shape[0]
    row0 = i + b + 1
    if row0 >= n:
        return
    A[row0:, i].zero_()
    A[i, row0:].zero_()


def sbr_to_band_triton_(A: torch.Tensor, b: int, *, block_n: int = 1024, bm: int = 64, bn: int = 64):
    """
    Stage 1: reduce symmetric A to symmetric band with lower bandwidth b (uplo=L).
    Prototype algorithm: for each i, form Householder on x = A[i+b:, i] to zero entries below i+b,
    then apply H to A12 (left-only) and A22 (two-sided) using Triton kernels.

    This keeps A explicitly symmetric (updates both lower/upper blocks).
    """
    assert A.is_cuda and A.ndim == 2 and A.shape[0] == A.shape[1]
    assert A.dtype == torch.float64
    n = A.shape[0]
    assert b >= 1 and b < n

    # Work buffers (max sizes).
    v_buf = torch.empty((n,), device=A.device, dtype=A.dtype)
    w_buf = torch.empty((n,), device=A.device, dtype=A.dtype)
    y_buf = torch.empty((n,), device=A.device, dtype=A.dtype)

    for i in range(0, n - b - 1):
        base = i + b
        m = n - base
        if m <= 1:
            break

        # x := A[base:, i]
        # Clone so subsequent in-place updates don't affect reflector construction.
        x = A[base:, i].clone()
        v, tau, beta = _householder_vec(x)

        # Avoid host sync on tau; for tau==0 (rare) the update is a no-op anyway.

        # Copy v to a fixed buffer view (avoid reallocs in kernels).
        v_buf[:m].copy_(v)
        v_full = v_buf[:m]

        # Update the "off-band" block directly adjacent to the panel from the left only.
        # IMPORTANT: do NOT update columns <= i (already reduced), otherwise we reintroduce fill-in
        # outside the desired band. Only update columns (i+1 : base), which has width (b-1).
        col0 = i + 1
        col1 = base
        p = col1 - col0
        if p > 0:
            A12 = A[base:, col0:col1]  # (m, b-1)
            y = y_buf[:p]
            y.zero_()
            grid = (triton.cdiv(m, 256), triton.cdiv(p, 32))
            gemv_t_partial_kernel[grid](
                A12, v_full, y,
                M=m, N=p,
                stride_am=A12.stride(0), stride_an=A12.stride(1),
                stride_v=v_full.stride(0), stride_y=y.stride(0),
                BLOCK_M=256, BLOCK_N=32,
                num_warps=4,
            )
            grid2 = (triton.cdiv(m, 128), triton.cdiv(p, 32))
            rank1_update_rect_tau_ptr_kernel[grid2](
                A12, v_full, y, tau,
                M=m, N=p,
                stride_am=A12.stride(0), stride_an=A12.stride(1),
                stride_v=v_full.stride(0), stride_y=y.stride(0),
                BM=128, BN=32,
                num_warps=4,
            )
            # Mirror to keep explicit symmetry: A[col0:col1, base:] = A12^T
            grid3 = (triton.cdiv(m, 128), triton.cdiv(p, 32))
            copy_transpose_kernel[grid3](
                A12, A[col0:col1, base:],
                M=m, N=p,
                stride_sm=A12.stride(0), stride_sn=A12.stride(1),
                stride_dm=A[col0:col1, base:].stride(0), stride_dn=A[col0:col1, base:].stride(1),
                BM=128, BN=32,
                num_warps=4,
            )

        # Update A22 = A[base:, base:] two-sided using standard symmetric formula:
        # w = tau * A22 @ v
        A22 = A[base:, base:]
        w = w_buf[:m]
        w.zero_()
        grid4 = (m, triton.cdiv(m, block_n))
        matvec_partial_kernel[grid4](
            A22, v_full, w,
            M=m, N=m,
            stride_am=A22.stride(0), stride_an=A22.stride(1),
            stride_x=v_full.stride(0), stride_y=w.stride(0),
            BLOCK_N=block_n,
            num_warps=4,
        )
        w *= tau

        # alpha = -0.5 * tau * (v^T w); w = w + alpha*v
        alpha = (-0.5) * tau * triton_dot(v_full, w)
        w += alpha * v_full

        # A22 -= v w^T + w v^T
        grid5 = (triton.cdiv(m, bm), triton.cdiv(m, bn))
        rank2_update_kernel[grid5](
            A22, v_full, w,
            M=m, N=m,
            stride_am=A22.stride(0), stride_an=A22.stride(1),
            stride_v=v_full.stride(0), stride_w=w.stride(0),
            BM=bm, BN=bn,
            num_warps=4,
        )

        # Finally, write the band element and clear entries below the band in col/row i.
        A[base, i] = beta
        A[i, base] = beta
        _zero_below_band_(A, i, b)


def _band_fill_from_dense_lower_(A_band: torch.Tensor, B: torch.Tensor, b: int):
    """Fill MAGMA-like lower band storage A_band (2*b+1,n) from dense B (n,n), using only the lower b-band."""
    assert A_band.ndim == 2 and B.ndim == 2 and B.shape[0] == B.shape[1]
    n = B.shape[0]
    lda = A_band.shape[0]
    assert lda == 2 * b + 1
    assert A_band.shape[1] == n
    A_band.zero_()
    # A_band[diff, j] = B[j+diff, j] for diff=0..b
    for j in range(n):
        mmax = min(b, n - 1 - j)
        if mmax >= 0:
            A_band[0:mmax + 1, j].copy_(B[j:j + mmax + 1, j])


@triton.jit
def pack_band_lower_kernel(B_ptr, A_band_ptr,
                           n: tl.constexpr,
                           b: tl.constexpr,
                           stride_b0, stride_b1,
                           stride_a0, stride_a1,
                           BLOCK_N: tl.constexpr,
                           BLOCK_D: tl.constexpr):
    """
    Pack dense symmetric-lower band into MAGMA band storage:
      A_band[diff, j] = B[j+diff, j], diff=0..b.
    A_band has shape (2*b+1, n); we only fill rows 0..b (lower band).
    """
    pid_j = tl.program_id(0)
    pid_d = tl.program_id(1)
    j = pid_j * BLOCK_N + tl.arange(0, BLOCK_N)
    d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_j = j < n
    mask_d = d <= b
    row = j[None, :] + d[:, None]
    mask = mask_d[:, None] & mask_j[None, :] & (row < n)

    val = tl.load(B_ptr + row * stride_b0 + j[None, :] * stride_b1, mask=mask, other=0.0).to(tl.float64)
    tl.store(A_band_ptr + d[:, None] * stride_a0 + j[None, :] * stride_a1, val, mask=mask)


def _band_fill_from_dense_lower_triton_(A_band: torch.Tensor, B: torch.Tensor, b: int):
    """Triton version of _band_fill_from_dense_lower_ (much faster for large n)."""
    assert A_band.ndim == 2 and B.ndim == 2 and B.shape[0] == B.shape[1]
    n = B.shape[0]
    lda = A_band.shape[0]
    assert lda == 2 * b + 1
    assert A_band.shape[1] == n
    A_band.zero_()
    grid = (triton.cdiv(n, 256), triton.cdiv(b + 1, 32))
    pack_band_lower_kernel[grid](
        B, A_band,
        n=n, b=b,
        stride_b0=B.stride(0), stride_b1=B.stride(1),
        stride_a0=A_band.stride(0), stride_a1=A_band.stride(1),
        BLOCK_N=256, BLOCK_D=32,
        num_warps=4,
    )


def _larfy_band_lower_(A_band: torch.Tensor, col0: int, v: torch.Tensor, tau: torch.Tensor, *,
                       bm: int = 32, bn: int = 32):
    """
    Apply symmetric Householder to the diagonal block starting at (col0,col0) in band storage:
      A := (I - tau v v^T) A (I - tau v v^T)
    where v has length N and A is symmetric (stored lower).
    """
    n = v.numel()
    if n <= 1:
        return
    stride0, stride1 = A_band.stride(0), A_band.stride(1)
    # Fused small-block update (N <= nb <= 64 typically).
    band_larfy_lower_fused_kernel[(1,)](
        A_band, v, tau,
        N=n,
        stride0=stride0, stride1=stride1,
        col0=col0,
        BLK=64,
        num_warps=4,
    )


def _larfx_right_band_(A_band: torch.Tensor, row0: int, col0: int, M: int, N: int,
                       v: torch.Tensor, tau: torch.Tensor):
    """Apply right Householder: C := C * (I - tau v v^T) on C=A[row0:row0+M, col0:col0+N] in band storage."""
    if M <= 0 or N <= 0:
        return
    stride0, stride1 = A_band.stride(0), A_band.stride(1)
    # Fused: y = C v; C -= tau y v^T
    band_larfx_right_fused_kernel[(1,)](
        A_band, v, tau,
        M=M, N=N,
        stride0=stride0, stride1=stride1,
        row0=row0, col0=col0,
        BM=64, BN=64,
        num_warps=4,
    )


def _larfx_left_band_(A_band: torch.Tensor, row0: int, col0: int, M: int, N: int,
                      v: torch.Tensor, tau: torch.Tensor):
    """Apply left Householder: C := (I - tau v v^T) * C on C=A[row0:row0+M, col0:col0+N] in band storage."""
    if M <= 0 or N <= 0:
        return
    stride0, stride1 = A_band.stride(0), A_band.stride(1)
    # Fused: y = v^T C; C -= tau v y
    band_larfx_left_fused_kernel[(1,)](
        A_band, v, tau,
        M=M, N=N,
        stride0=stride0, stride1=stride1,
        row0=row0, col0=col0,
        BM=64, BN=64,
        num_warps=4,
    )


def _hbtype1_real_(n: int, nb: int, A_band: torch.Tensor, V: torch.Tensor, TAU: torch.Tensor,
                   st: int, ed: int, sweep: int, *, mod_store: int = 8):
    """Real equivalent of MAGMA hbtype1cb on band storage (wantz=0). Indices are 0-based."""
    assert 0 <= st <= ed < n
    assert st >= 1  # eliminates column (st-1)
    len_ = ed - st + 1
    vpos = (sweep % mod_store) * n + st
    taupos = (sweep % mod_store) * n + st
    V[vpos] = 1.0

    col = st - 1
    # Copy A(st+1:ed, col) into V and zero it.
    if len_ > 1:
        V[vpos + 1:vpos + len_].copy_(A_band[2:len_ + 1, col])
        A_band[2:len_ + 1, col].zero_()

    # Build Householder for x = [A(st,col); copied_tail]
    x = torch.empty((len_,), device=A_band.device, dtype=A_band.dtype)
    x[0] = A_band[1, col]  # A(st,st-1)
    if len_ > 1:
        x[1:].copy_(V[vpos + 1:vpos + len_])
    v, tau, beta = _householder_vec(x)
    V[vpos:vpos + len_].copy_(v)
    TAU[taupos] = tau
    A_band[1, col] = beta

    _larfy_band_lower_(A_band, st, V[vpos:vpos + len_], TAU[taupos])


def _hbtype3_real_(n: int, nb: int, A_band: torch.Tensor, V: torch.Tensor, TAU: torch.Tensor,
                   st: int, ed: int, sweep: int, *, mod_store: int = 8):
    """Real equivalent of MAGMA hbtype3cb on band storage (wantz=0). Indices are 0-based."""
    assert 0 <= st <= ed < n
    len_ = ed - st + 1
    vpos = (sweep % mod_store) * n + st
    taupos = (sweep % mod_store) * n + st
    _larfy_band_lower_(A_band, st, V[vpos:vpos + len_], TAU[taupos])


def _hbtype2_real_(n: int, nb: int, A_band: torch.Tensor, V: torch.Tensor, TAU: torch.Tensor,
                   st: int, ed: int, sweep: int, *, mod_store: int = 8):
    """Real equivalent of MAGMA hbtype2cb on band storage (wantz=0). Indices are 0-based."""
    assert 0 <= st <= ed < n
    vpos = (sweep % mod_store) * n + st
    taupos = (sweep % mod_store) * n + st

    J1 = ed + 1
    J2 = min(ed + nb, n - 1)
    len_ = ed - st + 1
    lem = J2 - J1 + 1
    if lem > 0:
        _larfx_right_band_(A_band, J1, st, lem, len_, V[vpos:vpos + len_], TAU[taupos])

    if lem > 1:
        vpos2 = (sweep % mod_store) * n + J1
        taupos2 = (sweep % mod_store) * n + J1
        V[vpos2] = 1.0

        # Copy bulge column below the first element into V and zero it.
        # Vector is A(J1+1:J2, st).
        start = (J1 + 1) - st
        end = J2 - st
        V[vpos2 + 1:vpos2 + lem].copy_(A_band[start:end + 1, st])
        A_band[start:end + 1, st].zero_()

        x = torch.empty((lem,), device=A_band.device, dtype=A_band.dtype)
        x[0] = A_band[J1 - st, st]
        x[1:].copy_(V[vpos2 + 1:vpos2 + lem])
        v, tau, beta = _householder_vec(x)
        V[vpos2:vpos2 + lem].copy_(v)
        TAU[taupos2] = tau
        A_band[J1 - st, st] = beta

        # Apply left update on A(J1:J2, st+1:ed)
        len2 = len_ - 1
        if len2 > 0:
            _larfx_left_band_(A_band, J1, st + 1, lem, len2, V[vpos2:vpos2 + lem], TAU[taupos2])


def bc_band_to_tridiag_magma_like_(B: torch.Tensor, b: int, *, grsiz: int = 4, fused_hbtypes: bool = True):
    """
    Stage 2 (BC): MAGMA-like bulge chasing on a band matrix.

    Input B is dense symmetric (lower band with bandwidth b).
    Internally converts to MAGMA band storage (2*b+1, n) and runs a serial schedule
    equivalent to MAGMA's hb2st bulge chasing (wantz=0).

    Returns (d, e) where d is diag and e is subdiag of the final tridiagonal.
    """
    assert B.is_cuda and B.ndim == 2 and B.shape[0] == B.shape[1]
    assert B.dtype == torch.float64
    n = B.shape[0]
    nb = b
    if fused_hbtypes:
        # The fused hbtype kernels currently assume nb <= 64 (BLK=64) to keep register usage bounded.
        # Extend by parameterizing BLK if you want larger bands.
        assert nb <= 64, f"fused_hbtypes requires b<=64, got b={nb}"
    lda = 2 * nb + 1

    # Fortran-like strides (stride0=1, stride1=lda) via transpose view.
    storage = torch.empty((n, lda), device=B.device, dtype=B.dtype)
    A_band = storage.T
    _band_fill_from_dense_lower_(A_band, B, nb)

    # wantz=0 storage: vpos=(sweep%2)*n+st.
    mod_store = 2
    V = torch.zeros((mod_store * n,), device=B.device, dtype=B.dtype)
    TAU = torch.zeros((mod_store * n,), device=B.device, dtype=B.dtype)
    W = torch.empty((mod_store * n,), device=B.device, dtype=B.dtype)

    # Exact translation of MAGMA's magma_?tile_bulge_parallel with cores_num=1, wantz=0.
    # Key point: do NOT reorder tasks; the per-thread loop order is part of the synchronization contract.
    cores_num = 1
    my_core_id = 0
    assert grsiz >= 1
    shift = 3

    colblktile = 1 if grsiz == 1 else max(1, grsiz // 2)
    nbtiles = triton.cdiv(n - 1, nb)
    maxrequiredcores = max(nbtiles // colblktile, 1)
    colpercore = colblktile * nb
    allcoresnb = min(cores_num, maxrequiredcores)

    # Progress table: MAGMA uses 2*nbtiles + shift + cores_num + 10 (roughly). We oversize slightly.
    prog_len = 2 * nbtiles + shift + allcoresnb + 64
    prog = [0 for _ in range(prog_len)]

    stepercol = (shift + grsiz - 1) // grsiz
    thgrsiz = n
    thgrnb = triton.cdiv(n - 1, thgrsiz)

    def _cond_wait(idx: int, want: int):
        # CPU spin (serial schedule should satisfy without long waits).
        t0 = time.time()
        while prog[idx] != want:
            if (time.time() - t0) > 2.0:
                raise RuntimeError(f"hb2st cond_wait timeout: prog[{idx}]={prog[idx]} want {want}")

    def _cond_set(idx: int, val: int):
        prog[idx] = val

    for thgrid in range(1, thgrnb + 1):
        stt = (thgrid - 1) * thgrsiz + 1  # 1-based
        thed = min(stt + thgrsiz - 1, n - 1)
        i = stt
        while i <= (n - 1):
            ed = min(i, thed)
            if stt > ed:
                break
            for m in range(1, stepercol + 1):
                st = stt
                for sweepid in range(st, ed + 1):
                    blklastind = 0
                    for k in range(1, grsiz + 1):
                        myid = (i - sweepid) * (stepercol * grsiz) + (m - 1) * grsiz + k
                        if (myid % 2) == 0:
                            colpt = (myid // 2) * nb + 1 + sweepid - 1
                            stind = colpt - nb + 1
                            edind = min(colpt, n)
                            blklastind = colpt
                        else:
                            colpt = ((myid + 1) // 2) * nb + 1 + sweepid - 1
                            stind = colpt - nb + 1
                            edind = min(colpt, n)
                            if (stind >= edind - 1) and (edind == n):
                                blklastind = n
                            else:
                                blklastind = 0

                        coreid = (stind // colpercore) % max(allcoresnb, 1)
                        if my_core_id != coreid:
                            continue

                        st0 = stind - 1
                        ed0 = edind - 1
                        sw0 = sweepid - 1

                        if myid == 1:
                            _cond_wait(myid + shift - 1, sweepid - 1)
                            if fused_hbtypes:
                                # Fused: build reflector + diagonal block update in one launch.
                                hbtype1_real_fused_kernel[(1,)](
                                    A_band, V, TAU, W,
                                    n, nb,
                                    st0, ed0, sw0,
                                    A_band.stride(0), A_band.stride(1),
                                    MOD_STORE=mod_store, BLK=64,
                                    num_warps=4,
                                )
                            else:
                                _hbtype1_real_(n, nb, A_band, V, TAU, st0, ed0, sw0, mod_store=mod_store)
                            _cond_set(myid, sweepid)
                            if blklastind >= (n - 1):
                                for j2 in range(1, shift + 1):
                                    _cond_set(myid + j2, sweepid)
                        else:
                            _cond_wait(myid - 1, sweepid)
                            _cond_wait(myid + shift - 1, sweepid - 1)
                            if (myid % 2) == 0:
                                if fused_hbtypes:
                                    hbtype2_real_fused_kernel[(1,)](
                                        A_band, V, TAU,
                                        n, nb,
                                        st0, ed0, sw0,
                                        A_band.stride(0), A_band.stride(1),
                                        MOD_STORE=mod_store, BLK=64,
                                        num_warps=4,
                                    )
                                else:
                                    _hbtype2_real_(n, nb, A_band, V, TAU, st0, ed0, sw0, mod_store=mod_store)
                            else:
                                if fused_hbtypes:
                                    hbtype3_real_fused_kernel[(1,)](
                                        A_band, V, TAU, W,
                                        n, nb,
                                        st0, ed0, sw0,
                                        A_band.stride(0), A_band.stride(1),
                                        MOD_STORE=mod_store, BLK=64,
                                        num_warps=4,
                                    )
                                else:
                                    _hbtype3_real_(n, nb, A_band, V, TAU, st0, ed0, sw0, mod_store=mod_store)
                            _cond_set(myid, sweepid)
                            if blklastind >= (n - 1):
                                for j2 in range(1, shift + allcoresnb + 1):
                                    _cond_set(myid + j2, sweepid)

                        if blklastind >= (n - 1):
                            stt += 1
                            break
            i += 1

    d = A_band[0, :].clone()
    e = A_band[1, :n - 1].clone()
    return d, e


@triton.jit
def _prog_wait_eq(ptr, idx, want):
    # Simple spin-wait on a global progress table.
    # NOTE: This is safe only when the number of resident programs is limited (wave-based launch),
    # otherwise it can deadlock by occupying all SMs with spinning CTAs.
    #
    # IMPORTANT: When using prog as a "publish point" for other global stores (A_band, V, TAU),
    # we need acquire semantics on the read, otherwise another program may observe prog updated
    # before the producer's data writes become visible.
    while tl.atomic_add(ptr + idx, 0, sem="acquire").to(tl.int32) != want:
        pass


@triton.jit
def bc_magma_persistent_kernel(A_ptr, V_ptr, TAU_ptr, W_ptr, prog_ptr,
                               n, nb,
                               stride0, stride1,
                               grsiz: tl.constexpr,
                               MOD_STORE: tl.constexpr,
                               BLK: tl.constexpr):
    """
    GPU-resident version of MAGMA's tile bulge chasing scheduler (wantz=0),
    keeping the *same per-core loop order* to avoid deadlock/correctness issues.

    One Triton program ~= one MAGMA "core" thread (my_core_id).
    """
    my_core_id = tl.program_id(0).to(tl.int32)
    cores_num = tl.num_programs(0).to(tl.int32)

    shift = tl.full((), 3, tl.int32)
    nbtiles = (n - 2 + nb) // nb  # ceil((n-1)/nb)
    colblktile = tl.where(grsiz == 1, 1, grsiz // 2).to(tl.int32)
    maxrequiredcores = tl.maximum(nbtiles // colblktile, 1).to(tl.int32)
    colpercore = (colblktile * nb).to(tl.int32)
    allcoresnb = tl.minimum(cores_num, maxrequiredcores).to(tl.int32)

    # Only the first allcoresnb programs participate.
    if my_core_id >= allcoresnb:
        return

    # stepercol = ceil(shift/grsiz)
    i_tmp = shift // grsiz
    stepercol = tl.where(i_tmp * grsiz == shift, i_tmp, i_tmp + 1).to(tl.int32)

    # thgrsiz=n => thgrnb=1, so we drop thgrid.
    stt = tl.full((), 1, tl.int32)      # 1-based
    thed = (n - 1).to(tl.int32)         # 1-based max sweep

    i = stt
    while i <= (n - 1):
        ed = tl.minimum(i, thed)
        if stt > ed:
            # Equivalent to "break" out of the i loop.
            i = n
        else:
            m = tl.full((), 1, tl.int32)
            while m <= stepercol:
                st = stt
                sweepid = st
                while sweepid <= ed:
                    blklastind = tl.full((), 0, tl.int32)
                    k = tl.full((), 1, tl.int32)
                    while k <= grsiz:
                        myid = (i - sweepid) * (stepercol * grsiz) + (m - 1) * grsiz + k
                        # Compute (stind, edind, blklastind) in 1-based indexing.
                        if (myid % 2) == 0:
                            colpt = (myid // 2) * nb + 1 + sweepid - 1
                            stind = colpt - nb + 1
                            edind = tl.minimum(colpt, n)
                            blklastind = colpt
                        else:
                            colpt = ((myid + 1) // 2) * nb + 1 + sweepid - 1
                            stind = colpt - nb + 1
                            edind = tl.minimum(colpt, n)
                            blklastind = tl.where(((stind >= (edind - 1)) & (edind == n)), n, 0)

                        coreid = ((stind // colpercore) % allcoresnb).to(tl.int32)
                        if my_core_id == coreid:
                            st0 = stind - 1
                            ed0 = edind - 1
                            sw0 = sweepid - 1

                            if myid == 1:
                                _prog_wait_eq(prog_ptr, myid + shift - 1, sweepid - 1)
                                hbtype1_real_fused_body(
                                    A_ptr, V_ptr, TAU_ptr, W_ptr,
                                    n, nb,
                                    st0, ed0, sw0,
                                    stride0, stride1,
                                    MOD_STORE=MOD_STORE, BLK=BLK,
                                )
                                tl.atomic_xchg(prog_ptr + myid, sweepid, sem="release")
                                if blklastind >= (n - 1):
                                    j2 = tl.full((), 1, tl.int32)
                                    while j2 <= shift:
                                        tl.atomic_xchg(prog_ptr + (myid + j2), sweepid, sem="release")
                                        j2 += 1
                            else:
                                _prog_wait_eq(prog_ptr, myid - 1, sweepid)
                                _prog_wait_eq(prog_ptr, myid + shift - 1, sweepid - 1)
                                if (myid % 2) == 0:
                                    hbtype2_real_fused_body(
                                        A_ptr, V_ptr, TAU_ptr,
                                        n, nb,
                                        st0, ed0, sw0,
                                        stride0, stride1,
                                        MOD_STORE=MOD_STORE, BLK=BLK,
                                    )
                                else:
                                    hbtype3_real_fused_body(
                                        A_ptr, V_ptr, TAU_ptr, W_ptr,
                                        n, nb,
                                        st0, ed0, sw0,
                                        stride0, stride1,
                                        MOD_STORE=MOD_STORE, BLK=BLK,
                                    )
                                tl.atomic_xchg(prog_ptr + myid, sweepid, sem="release")
                                if blklastind >= (n - 1):
                                    j2 = tl.full((), 1, tl.int32)
                                    while j2 <= (shift + allcoresnb):
                                        tl.atomic_xchg(prog_ptr + (myid + j2), sweepid, sem="release")
                                        j2 += 1

                        # Update stt when reaching end-of-band marker.
                        if blklastind >= (n - 1):
                            stt += 1
                            k = grsiz + 1  # exit k loop
                        else:
                            k += 1
                    sweepid += 1
                m += 1
        i += 1


def bc_band_to_tridiag_magma_persistent_gpu_(B: torch.Tensor, b: int, *, grsiz: int = 4, workers: int | None = None, mod_store: int | None = None):
    """
    Stage 2 (BC): MAGMA-like hb2st bulge chasing, but with GPU-resident scheduling.

    This uses a single long-running Triton kernel and a global progress table (prog) for synchronization,
    eliminating Python per-task overhead and per-task kernel launches.
    """
    assert B.is_cuda and B.ndim == 2 and B.shape[0] == B.shape[1]
    assert B.dtype == torch.float64
    n = B.shape[0]
    nb = b
    assert nb <= 64, "persistent kernel currently assumes b<=64"
    assert grsiz >= 1

    lda = 2 * nb + 1
    storage = torch.empty((n, lda), device=B.device, dtype=B.dtype)
    A_band = storage.T
    _band_fill_from_dense_lower_triton_(A_band, B, nb)

    # Allocate enough progress slots for the largest myid used by the schedule.
    # The original MAGMA code can allocate a smaller table because it relies on implicit bounds from the nested loops,
    # but for safety (and to avoid OOB reads/writes), we allocate the worst-case size.
    colblktile = 1 if grsiz == 1 else max(1, grsiz // 2)
    nbtiles = triton.cdiv(n - 1, nb)
    maxrequiredcores = max(nbtiles // colblktile, 1)
    if workers is None:
        props = torch.cuda.get_device_properties(B.device)
        workers = int(props.multi_processor_count)
    workers = max(1, min(int(workers), maxrequiredcores))

    # Ring-buffer depth for V/TAU/W storage.
    # IMPORTANT: MOD_STORE=2 is fine for the serial schedule, but NOT safe once we have multiple
    # cores running concurrently (different sweeps can overlap and clobber the same slot).
    # We want MOD_STORE > maximum sweep distance between concurrently executing tasks.
    if mod_store is None or int(mod_store) <= 0:
        target = int(3 + workers + 2 * max(1, grsiz) + 8)  # shift + workers + slack
        mod_store = 1
        while mod_store < target:
            mod_store *= 2
    mod_store = int(mod_store)
    assert mod_store >= 4
    assert mod_store <= 4096
    V = torch.zeros((mod_store * n,), device=B.device, dtype=B.dtype)
    TAU = torch.zeros((mod_store * n,), device=B.device, dtype=B.dtype)
    W = torch.empty((mod_store * n,), device=B.device, dtype=B.dtype)

    shift = 3
    stepercol = (shift + grsiz - 1) // grsiz
    # Upper bound: myid <= (n-1) * (stepercol*grsiz) + grsiz
    prog_len = (n - 1) * (stepercol * grsiz) + (shift + workers + 128)
    prog = torch.zeros((prog_len,), device=B.device, dtype=torch.int32)

    bc_magma_persistent_kernel[(workers,)](
        A_band, V, TAU, W, prog,
        n, nb,
        A_band.stride(0), A_band.stride(1),
        grsiz=grsiz,
        MOD_STORE=mod_store,
        BLK=64,
        num_warps=4,
    )

    d = A_band[0, :].clone()
    e = A_band[1, :n - 1].clone()
    return d, e


@triton.jit
def bc_pipeline_worker_kernel(A_ptr, gCom_ptr,
                              n, nb,
                              stride0, stride1,
                              BLK: tl.constexpr):
    """
    GPU bulge chasing, pipeline-style (paper Algorithm 2 flavor), using MAGMA-like hbtype1 + (hbtype2+hbtype3)*.

    - We launch a small number of *worker* programs (typically ~= #SMs). Each worker processes
      sweep indices in a strided loop: i = pid, pid+workers, ...
    - Each sweep i maintains gCom[i] = current working row (0-based start index of the current diagonal block).
    - Before advancing a step, sweep i waits until gCom[i-1] is at least 2*nb ahead (paper uses 2*b).
    - The update itself follows the same band-storage semantics as our hbtype kernels:
        hbtype1: build reflector from column (st-1) and update diagonal block at st.
        hbtype2: right-update below-diagonal block, generate bulge reflector at (J1, st), left-update.
        hbtype3: apply symmetric update on next diagonal block at st=J1.

    Assumes MAGMA band storage layout in A_ptr with lda=2*nb+1 and storing lower triangle (diff=row-col >= 0).
    """
    pid = tl.program_id(0).to(tl.int32)
    workers = tl.num_programs(0).to(tl.int32)

    offs = tl.arange(0, BLK)
    i_idx = pid

    while i_idx < (n - 2):

        # Sweep index and initial working row.
        st = i_idx + 1  # 0-based
        tl.atomic_xchg(gCom_ptr + i_idx, st)

        # Compute hbtype1 at (st, ed) to initialize v/tau for this sweep.
        # Block length is min(nb, n-st).
        len_ = tl.minimum(nb, n - st).to(tl.int32)
        # Default reflector state (only meaningful if len_ > 1).
        tau = tl.full((), 0.0, tl.float64)
        v = tl.zeros([BLK], tl.float64)

        # Process this sweep if there's work to do.
        if len_ > 1:
            # Respect the same dependency rule before touching data.
            if i_idx != 0:
                while (st + 2 * nb) > tl.load(gCom_ptr + (i_idx - 1), cache_modifier=".cg").to(tl.int32):
                    pass

            col1 = st - 1
            # Load x = A(st:st+len_-1, col) from band storage: diff=(row-col) = 1..len_
            mask_len = offs < len_
            x = tl.load(A_ptr + (offs + 1) * stride0 + col1 * stride1, mask=mask_len, other=0.0).to(tl.float64)
            alpha = tl.load(A_ptr + 1 * stride0 + col1 * stride1).to(tl.float64)
            xt = tl.where(offs > 0, x, 0.0)
            sigma = tl.sum(xt * xt, axis=0)
            is_zero = sigma == 0.0
            norm = tl.sqrt(alpha * alpha + sigma)
            beta0 = tl.where(alpha <= 0.0, norm, -norm)
            beta = tl.where(is_zero, alpha, beta0)
            tau0 = (beta0 - alpha) / beta0
            tau = tl.where(is_zero, 0.0, tau0)
            scale = tl.where(is_zero, 0.0, 1.0 / (alpha - beta0))
            v = tl.where(offs == 0, 1.0, x * scale).to(tl.float64)

            # Write beta and zero below in the current column.
            tl.store(A_ptr + 1 * stride0 + col1 * stride1, beta)
            mask_zero = (offs >= 1) & mask_len
            tl.store(A_ptr + (offs + 1) * stride0 + col1 * stride1, 0.0, mask=mask_zero)

            # Diagonal symmetric update on A(st:st+len_, st:st+len_) (stored lower).
            i0 = tl.arange(0, BLK)
            j0 = tl.arange(0, BLK)
            mi = i0 < len_
            mj = j0 < len_
            mask_blk = mi[:, None] & mj[None, :]
            ii = tl.maximum(i0[:, None], j0[None, :])
            jj = tl.minimum(i0[:, None], j0[None, :])
            diff = (ii - jj).to(tl.int32)
            Aij = tl.load(A_ptr + diff * stride0 + (st + jj) * stride1, mask=mask_blk, other=0.0).to(tl.float64)
            w = tl.sum(Aij * v[None, :], axis=1) * tau
            vt_w = tl.sum(v * w, axis=0)
            alpha2 = (-0.5) * tau * vt_w
            w = w + alpha2 * v
            Anew = Aij - v[:, None] * w[None, :] - w[:, None] * v[None, :]
            lower = i0[:, None] >= j0[None, :]
            mask_store = mask_blk & lower
            diff2 = (i0[:, None] - j0[None, :]).to(tl.int32)
            tl.store(A_ptr + diff2 * stride0 + (st + j0[None, :]) * stride1, Anew, mask=mask_store)

            # Main sweep loop: advance bulge down-right in steps of size nb (bandwidth).
            while st < (n - 1):
                # wait for separation from previous sweep (paper uses 2*b).
                if i_idx != 0:
                    while (st + 2 * nb) > tl.load(gCom_ptr + (i_idx - 1), cache_modifier=".cg").to(tl.int32):
                        pass

                # Current diagonal block length (may shrink near bottom).
                len_ = tl.minimum(nb, n - st).to(tl.int32)
                if len_ <= 1:
                    st = n  # terminate sweep loop
                else:
                    ed = st + len_ - 1
                    J1 = ed + 1
                    if J1 >= n:
                        st = n
                    else:
                        lem = tl.minimum(nb, n - J1).to(tl.int32)
                        # We need at least a 2-row block below to generate a new bulge reflector.
                        if lem <= 1:
                            st = n
                        else:
                            J2 = J1 + lem - 1

                            # ---- Bol: left update on the block to the left of the diagonal block (st:ed, st-wleft:st-1).
                            # This is required for a full similarity transform on the banded matrix.
                            wleft = tl.minimum(nb, st).to(tl.int32)
                            if wleft > 0:
                                offs_r = tl.arange(0, BLK)
                                offs_c = tl.arange(0, BLK)
                                mask_r = offs_r < len_
                                mask_c = offs_c < wleft
                                rowB = (st + offs_r)[:, None]
                                colB = (st - wleft + offs_c)[None, :]
                                maskB = mask_r[:, None] & mask_c[None, :]
                                diffB = (rowB - colB).to(tl.int32)
                                Bol = tl.load(A_ptr + diffB * stride0 + colB * stride1, mask=maskB, other=0.0).to(tl.float64)
                                yB = tl.sum(Bol * v[:, None], axis=0)  # (wleft,)
                                Bol = Bol - tau * (v[:, None] * yB[None, :])
                                tl.store(A_ptr + diffB * stride0 + colB * stride1, Bol, mask=maskB)

                            # ---- hbtype2: right update on C = A(J1:J2, st:ed) with current v/tau.
                            offs_m = tl.arange(0, BLK)
                            offs_k = tl.arange(0, BLK)
                            mask_m = offs_m < lem
                            mask_k = offs_k < len_
                            rowC = (J1 + offs_m)[:, None]
                            colC = (st + offs_k)[None, :]
                            maskC = mask_m[:, None] & mask_k[None, :]
                            diffC = (rowC - colC).to(tl.int32)
                            C = tl.load(A_ptr + diffC * stride0 + colC * stride1, mask=maskC, other=0.0).to(tl.float64)
                            y = tl.sum(C * v[None, :], axis=1)  # (lem,)
                            C = C - tau * (y[:, None] * v[None, :])
                            tl.store(A_ptr + diffC * stride0 + colC * stride1, C, mask=maskC)

                            # ---- generate bulge reflector from column st, rows J1:J2 (length lem).
                            offs_l = tl.arange(0, BLK)
                            mask_l = offs_l < lem
                            diff0 = (J1 - st) + offs_l
                            x2 = tl.load(A_ptr + diff0 * stride0 + st * stride1, mask=mask_l, other=0.0).to(tl.float64)
                            alpha_b = tl.load(A_ptr + (J1 - st) * stride0 + st * stride1).to(tl.float64)
                            xt2 = tl.where(offs_l > 0, x2, 0.0)
                            sigma2 = tl.sum(xt2 * xt2, axis=0)
                            is_zero2 = sigma2 == 0.0
                            norm2 = tl.sqrt(alpha_b * alpha_b + sigma2)
                            beta0_2 = tl.where(alpha_b <= 0.0, norm2, -norm2)
                            beta2 = tl.where(is_zero2, alpha_b, beta0_2)
                            tau0_2 = (beta0_2 - alpha_b) / beta0_2
                            tau2 = tl.where(is_zero2, 0.0, tau0_2)
                            scale2 = tl.where(is_zero2, 0.0, 1.0 / (alpha_b - beta0_2))
                            v2 = tl.where(offs_l == 0, 1.0, x2 * scale2).to(tl.float64)

                            # Write beta and clear below in column st (bulge column).
                            tl.store(A_ptr + (J1 - st) * stride0 + st * stride1, beta2)
                            mask_zero2 = (offs_l >= 1) & mask_l
                            tl.store(A_ptr + diff0 * stride0 + st * stride1, 0.0, mask=mask_zero2)

                            # ---- hbtype2: left update on L = A(J1:J2, st+1:ed) with v2/tau2.
                            len2 = len_ - 1
                            if len2 > 0:
                                offs_n = tl.arange(0, BLK)
                                mask_n = offs_n < len2
                                row2 = (J1 + offs_l)[:, None]
                                col2 = ((st + 1) + offs_n)[None, :]
                                maskL = mask_l[:, None] & mask_n[None, :]
                                diffL = (row2 - col2).to(tl.int32)
                                L = tl.load(A_ptr + diffL * stride0 + col2 * stride1, mask=maskL, other=0.0).to(tl.float64)
                                y2 = tl.sum(L * v2[:, None], axis=0)  # (len2,)
                                L = L - tau2 * (v2[:, None] * y2[None, :])
                                tl.store(A_ptr + diffL * stride0 + col2 * stride1, L, mask=maskL)

                            # ---- hbtype3: diagonal update on next block at st = J1 using v2/tau2.
                            st = J1
                            tau = tau2

                            len_ = lem
                            mask_len = offs < len_
                            v = tl.where(mask_len, v2, 0.0).to(tl.float64)

                            i0 = tl.arange(0, BLK)
                            j0 = tl.arange(0, BLK)
                            mi = i0 < len_
                            mj = j0 < len_
                            mask_blk = mi[:, None] & mj[None, :]
                            ii = tl.maximum(i0[:, None], j0[None, :])
                            jj = tl.minimum(i0[:, None], j0[None, :])
                            diff = (ii - jj).to(tl.int32)
                            Aij = tl.load(A_ptr + diff * stride0 + (st + jj) * stride1, mask=mask_blk, other=0.0).to(tl.float64)
                            w = tl.sum(Aij * v[None, :], axis=1) * tau
                            vt_w = tl.sum(v * w, axis=0)
                            alpha2 = (-0.5) * tau * vt_w
                            w = w + alpha2 * v
                            Anew = Aij - v[:, None] * w[None, :] - w[:, None] * v[None, :]
                            lower = i0[:, None] >= j0[None, :]
                            mask_store = mask_blk & lower
                            diff2 = (i0[:, None] - j0[None, :]).to(tl.int32)
                            tl.store(A_ptr + diff2 * stride0 + (st + j0[None, :]) * stride1, Anew, mask=mask_store)

                            # Publish progress *after* completing the update for this step.
                            tl.atomic_xchg(gCom_ptr + i_idx, st)

        # Mark sweep complete (also covers len_ <= 1 case).
        tl.atomic_xchg(gCom_ptr + i_idx, n + 2 * nb)

        i_idx += workers

    return


@triton.jit
def bc_pipeline_hbtypes_kernel(A_ptr, V_ptr, TAU_ptr, W_ptr, gCom_ptr,
                               n, nb,
                               stride0, stride1,
                               MOD_STORE: tl.constexpr,
                               BLK: tl.constexpr):
    """
    Paper-style sweep pipeline (Algorithm 2) using our hbtype1/2/3 primitives (wantz=0, uplo=L).

    Each program handles sweep indices in a strided loop: i = pid, pid+workers, ...
    gCom[i] tracks the current working row (st) for sweep i and enforces a 2*b separation to avoid conflicts.
    """
    pid = tl.program_id(0).to(tl.int32)
    workers = tl.num_programs(0).to(tl.int32)

    i_idx = pid
    while i_idx < (n - 2):
        st = i_idx + 1
        tl.atomic_xchg(gCom_ptr + i_idx, st, sem="release")

        first = tl.full((), 1, tl.int1)
        while st < (n - 1):
            if i_idx != 0:
                pred = tl.atomic_add(gCom_ptr + (i_idx - 1), 0, sem="acquire").to(tl.int32)
                while (st + 2 * nb) > pred:
                    pred = tl.atomic_add(gCom_ptr + (i_idx - 1), 0, sem="acquire").to(tl.int32)

            ed = tl.minimum(st + nb - 1, n - 1)
            if (ed - st) <= 0:
                st = n
            else:
                if first:
                    hbtype1_real_fused_body(
                        A_ptr, V_ptr, TAU_ptr, W_ptr,
                        n, nb,
                        st, ed, i_idx,
                        stride0, stride1,
                        MOD_STORE=MOD_STORE, BLK=BLK,
                    )
                    first = tl.full((), 0, tl.int1)
                else:
                    hbtype3_real_fused_body(
                        A_ptr, V_ptr, TAU_ptr, W_ptr,
                        n, nb,
                        st, ed, i_idx,
                        stride0, stride1,
                        MOD_STORE=MOD_STORE, BLK=BLK,
                    )

                hbtype2_real_fused_body(
                    A_ptr, V_ptr, TAU_ptr,
                    n, nb,
                    st, ed, i_idx,
                    stride0, stride1,
                    MOD_STORE=MOD_STORE, BLK=BLK,
                )

                st = ed + 1
                tl.atomic_xchg(gCom_ptr + i_idx, st, sem="release")

        tl.atomic_xchg(gCom_ptr + i_idx, n + 2 * nb, sem="release")
        i_idx += workers

    return


@triton.jit
def bc_pipeline_hbtypes_wave_kernel(A_ptr, V_ptr, TAU_ptr, W_ptr, gCom_ptr,
                                    bar_count_ptr, bar_epoch_ptr,
                                    n, nb,
                                    stride0, stride1,
                                    MOD_STORE: tl.constexpr,
                                    BLK: tl.constexpr):
    """
    Wave-based variant of the paper-style sweep pipeline.

    Rationale: a pure strided assignment (i = pid, pid+workers, ...) can lead to excessive spinning (or even
    stall) when a sweep starts before its predecessor sweep has even begun. Here we run sweeps in contiguous
    waves of size `workers`:
      wave_base = 0, workers, 2*workers, ...
      sweep = wave_base + pid
    and use a lightweight global barrier between waves.
    """
    pid = tl.program_id(0).to(tl.int32)
    workers = tl.num_programs(0).to(tl.int32)

    wave_base = tl.full((), 0, tl.int32)
    epoch = tl.full((), 0, tl.int32)

    # Initialize epoch on all programs (read once).
    epoch = tl.atomic_add(bar_epoch_ptr, 0, sem="acquire").to(tl.int32)

    while wave_base < (n - 2):
        sweep = wave_base + pid
        active = tl.minimum(workers, (n - 2) - wave_base)

        if sweep < (n - 2):
            st = sweep + 1
            tl.atomic_xchg(gCom_ptr + sweep, st, sem="release")

            first = tl.full((), 1, tl.int1)
            while st < (n - 1):
                if sweep != 0:
                    pred = tl.atomic_add(gCom_ptr + (sweep - 1), 0, sem="acquire").to(tl.int32)
                    while (st + 2 * nb) > pred:
                        pred = tl.atomic_add(gCom_ptr + (sweep - 1), 0, sem="acquire").to(tl.int32)

                ed = tl.minimum(st + nb - 1, n - 1)
                if (ed - st) <= 0:
                    st = n
                else:
                    if first:
                        hbtype1_real_fused_body(
                            A_ptr, V_ptr, TAU_ptr, W_ptr,
                            n, nb,
                            st, ed, sweep,
                            stride0, stride1,
                            MOD_STORE=MOD_STORE, BLK=BLK,
                        )
                        first = tl.full((), 0, tl.int1)
                    else:
                        hbtype3_real_fused_body(
                            A_ptr, V_ptr, TAU_ptr, W_ptr,
                            n, nb,
                            st, ed, sweep,
                            stride0, stride1,
                            MOD_STORE=MOD_STORE, BLK=BLK,
                        )

                    hbtype2_real_fused_body(
                        A_ptr, V_ptr, TAU_ptr,
                        n, nb,
                        st, ed, sweep,
                        stride0, stride1,
                        MOD_STORE=MOD_STORE, BLK=BLK,
                    )

                    st = ed + 1
                    tl.atomic_xchg(gCom_ptr + sweep, st, sem="release")

            tl.atomic_xchg(gCom_ptr + sweep, n + 2 * nb, sem="release")

        # ---- global barrier between waves ----
        if pid < active:
            tl.atomic_add(bar_count_ptr, 1, sem="release")
        if pid == 0:
            # Wait until all active programs have arrived.
            while tl.atomic_add(bar_count_ptr, 0, sem="acquire").to(tl.int32) != active:
                pass
            # Reset count and advance epoch.
            tl.atomic_xchg(bar_count_ptr, 0, sem="release")
            epoch_next = epoch + 1
            tl.atomic_xchg(bar_epoch_ptr, epoch_next, sem="release")
            epoch = epoch_next
        else:
            # Wait for epoch change.
            want = epoch + 1
            while tl.atomic_add(bar_epoch_ptr, 0, sem="acquire").to(tl.int32) != want:
                pass
            epoch = want

        wave_base += workers

    return


@triton.jit
def bc_pipeline_hbtypes_queue_kernel(A_ptr, V_ptr, TAU_ptr, W_ptr, gCom_ptr,
                                     next_sweep_ptr,
                                     n, nb,
                                     stride0, stride1,
                                     MOD_STORE: tl.constexpr,
                                     BLK: tl.constexpr):
    """
    Work-queue variant of the paper-style sweep pipeline.

    Each program repeatedly grabs the next sweep index from a global counter, which keeps the set of active sweeps
    roughly contiguous. This dramatically reduces pathological spinning compared to a fixed strided assignment
    (i = pid, pid+workers, ...), and avoids deadlock scenarios where a sweep waits on a predecessor sweep that
    hasn't started yet.

    Slotting: we use the program id as the 'sweep' argument to hbtype1/2/3 so slot = pid % MOD_STORE is stable.
    """
    pid = tl.program_id(0).to(tl.int32)

    done = tl.full((), 0, tl.int1)
    while done == 0:
        sweep = tl.atomic_add(next_sweep_ptr, 1, sem="acquire").to(tl.int32)
        if sweep >= (n - 2):
            done = tl.full((), 1, tl.int1)
        else:

            st = sweep + 1
            tl.atomic_xchg(gCom_ptr + sweep, st, sem="release")

            first = tl.full((), 1, tl.int1)
            while st < (n - 1):
                if sweep != 0:
                    pred = tl.atomic_add(gCom_ptr + (sweep - 1), 0, sem="acquire").to(tl.int32)
                    while (st + 2 * nb) > pred:
                        pred = tl.atomic_add(gCom_ptr + (sweep - 1), 0, sem="acquire").to(tl.int32)

                ed = tl.minimum(st + nb - 1, n - 1)
                if (ed - st) <= 0:
                    st = n
                else:
                    if first:
                        hbtype1_real_fused_body(
                            A_ptr, V_ptr, TAU_ptr, W_ptr,
                            n, nb,
                            st, ed, pid,  # slotting only
                            stride0, stride1,
                            MOD_STORE=MOD_STORE, BLK=BLK,
                        )
                        first = tl.full((), 0, tl.int1)
                    else:
                        hbtype3_real_fused_body(
                            A_ptr, V_ptr, TAU_ptr, W_ptr,
                            n, nb,
                            st, ed, pid,  # slotting only
                            stride0, stride1,
                            MOD_STORE=MOD_STORE, BLK=BLK,
                        )

                    hbtype2_real_fused_body(
                        A_ptr, V_ptr, TAU_ptr,
                        n, nb,
                        st, ed, pid,  # slotting only
                        stride0, stride1,
                        MOD_STORE=MOD_STORE, BLK=BLK,
                    )

                    st = ed + 1
                    tl.atomic_xchg(gCom_ptr + sweep, st, sem="release")

            tl.atomic_xchg(gCom_ptr + sweep, n + 2 * nb, sem="release")

    return


def bc_band_to_tridiag_pipeline_gpu_(B: torch.Tensor, b: int, *, workers: int | None = None):
    """
    Stage 2 (BC): GPU-based bulge chasing, following the paper's sweep-pipeline idea (gCom)
    but using our band hbtype algebra (wantz=0, uplo=L).

    This is meant to eliminate the Python-driven serial scheduling overhead in bc_magma.
    """
    assert B.is_cuda and B.ndim == 2 and B.shape[0] == B.shape[1]
    assert B.dtype == torch.float64
    n = B.shape[0]
    nb = b
    assert nb <= 64, "pipeline kernel currently assumes b<=64"

    lda = 2 * nb + 1
    storage = torch.empty((n, lda), device=B.device, dtype=B.dtype)
    A_band = storage.T
    _band_fill_from_dense_lower_triton_(A_band, B, nb)

    if workers is None:
        props = torch.cuda.get_device_properties(B.device)
        # Queue-based scheduler avoids the classic spin-deadlock of fixed assignment, so we can
        # safely default to the SM count for optimal throughput.
        workers = int(props.multi_processor_count)
    workers = max(1, min(workers, max(1, n - 2)))

    # Per-program slotting for V/TAU/W: MOD_STORE==workers ensures slot = sweep % workers = pid
    # for our strided sweep assignment (i = pid, pid+workers, ...), avoiding any ring-buffer hazards.
    V = torch.zeros((workers * n,), device=B.device, dtype=B.dtype)
    TAU = torch.zeros((workers * n,), device=B.device, dtype=B.dtype)
    W = torch.empty((workers * n,), device=B.device, dtype=B.dtype)

    # gCom: current working row for each sweep (int32).
    gCom = torch.zeros((n,), device=B.device, dtype=torch.int32)

    next_sweep = torch.zeros((1,), device=B.device, dtype=torch.int32)

    bc_pipeline_hbtypes_queue_kernel[(workers,)](
        A_band, V, TAU, W, gCom,
        next_sweep,
        n, nb,
        A_band.stride(0), A_band.stride(1),
        MOD_STORE=workers,
        BLK=64,
        num_warps=4,
        num_stages=1,
    )

    d = A_band[0, :].clone()
    e = A_band[1, :n - 1].clone()
    return d, e


def bc_band_to_tridiag_triton_(B: torch.Tensor, b: int, *, dense: bool = True, local_w: int | None = None):
    """
    Stage 2 (BC): reduce symmetric band (lower bandwidth b) to tridiagonal using bulge chasing.
    Prototype: symmetric band -> tridiagonal using Givens rotations (DSBTRD-like).

    - dense=True: correctness baseline; apply each similarity to full rows/cols and eliminate all
      entries below the first subdiagonal (O(n^3)).
    - dense=False: experimental local-chasing mode. It applies similarities only to a local window
      around the active rows/cols to avoid densifying the matrix. The window half-width is
      `local_w` (defaults to 2*b). This is *not yet* a MAGMA/LAPACK-grade bulge chase, but it is
      a practical stepping stone.
    """
    assert B.is_cuda and B.ndim == 2 and B.shape[0] == B.shape[1]
    assert B.dtype == torch.float64
    n = B.shape[0]
    assert b >= 1 and b < n

    stride0, stride1 = B.stride(0), B.stride(1)
    BLOCK = 1024
    cs = torch.empty((2,), device=B.device, dtype=B.dtype)

    if dense:
        for k in range(0, n - 2):
            i_max = n - 1
            for i in range(i_max, k + 1, -1):
                if i <= k + 1:
                    break
                givens_cs_kernel[(1,)](B, cs, i - 1, i, k, stride0, stride1)
                grid = (triton.cdiv(n, BLOCK),)
                apply_givens_rows_kernel[grid](B, cs, n, i - 1, i, stride0, stride1, BLOCK=BLOCK, num_warps=4)
                apply_givens_cols_kernel[grid](B, cs, n, i - 1, i, stride0, stride1, BLOCK=BLOCK, num_warps=4)
                set2_kernel[(1,)](B, i, k, stride0, stride1, v=0.0)
                set2_kernel[(1,)](B, k, i, stride0, stride1, v=0.0)
    else:
        # Band-limited bulge chasing with plane rotations.
        # Maintain an active bandwidth of (b+1): a single extra subdiagonal "bulge"
        # is chased to the bottom-right. Updates are restricted to the minimal range
        # that can be affected given band structure and the bulge position.
        w = (b + 1) if local_w is None else int(local_w)
        w = max(b + 1, min(w, n - 1))
        for k in range(0, n - 2):
            # Only rows within the current active bandwidth can have nonzeros in column k.
            i_max = min(n - 1, k + w)
            for i in range(i_max, k + 1, -1):
                if i <= k + 1:
                    break
                givens_cs_kernel[(1,)](B, cs, i - 1, i, k, stride0, stride1)

                # Similarity: A <- G^T A G.
                # Due to bandedness, rotating rows/cols (i-1,i) can only affect columns/rows
                # from the current elimination column k up to the right edge of those rows.
                # Include k-1 to correctly propagate the "bulge" that gets created
                # one column to the left during the similarity update.
                start = 0 if k == 0 else (k - 1)
                end = min(n, i + w + 1)  # exclusive
                count = end - start
                if count > 0:
                    grid = (triton.cdiv(count, BLOCK),)
                    apply_givens_rows_range_kernel[grid](
                        B, cs, start, count, i - 1, i, stride0, stride1,
                        BLOCK=BLOCK, num_warps=4,
                    )
                    apply_givens_cols_range_kernel[grid](
                        B, cs, start, count, i - 1, i, stride0, stride1,
                        BLOCK=BLOCK, num_warps=4,
                    )
                set2_kernel[(1,)](B, i, k, stride0, stride1, v=0.0)
                set2_kernel[(1,)](B, k, i, stride0, stride1, v=0.0)


def bc_band_to_tridiag_householder_(B: torch.Tensor, b: int, *,
                                    block_n: int = 128):
    """
    Stage 2 (BC) correctness-first: band -> tridiagonal using *sequential* Householder similarity,
    exploiting band locality.

    This is not the paper's parallel sweep pipeline yet, but it gives a correct band-to-tridiag
    reduction that only touches a local trailing neighborhood.

    Assumptions:
      - B is symmetric (dense storage) and initially has half-bandwidth <= b (uplo='L').
      - During chasing, fill-in can temporarily grow up to ~ (window_scale*b).
    """
    assert B.is_cuda and B.ndim == 2 and B.shape[0] == B.shape[1]
    assert B.dtype == torch.float64
    n = B.shape[0]
    assert b >= 1 and b < n
    # We'll update a window that covers the entire interaction region for a b-banded symmetric
    # matrix under a reflector of length b:
    #   columns/rows in [k-b, k+2b+1] can be affected.
    #
    # This is the key correctness detail that the earlier "window_scale*b starting at k"
    # missed (it ignored the left band), which breaks similarity and eigenvalues.
    #
    # Window width <= 3*b + 2.
    v_buf = torch.empty((b,), device=B.device, dtype=B.dtype)
    y_buf = torch.empty((3 * b + 2,), device=B.device, dtype=B.dtype)
    z_buf = torch.empty((3 * b + 2,), device=B.device, dtype=B.dtype)

    for k in range(0, n - 2):
        m = min(b, n - k - 1)
        if m <= 1:
            continue

        r0 = k + 1
        r1 = r0 + m

        x = B[r0:r1, k].clone()
        v, tau, beta = _householder_vec(x)
        # Avoid host sync; if tau==0 this step is a no-op anyway.

        v_buf[:m].copy_(v)
        v = v_buf[:m]

        # Apply H^T to the panel column (annihilate below the first entry) and keep symmetry.
        B[r0, k] = beta
        B[k, r0] = beta
        if m > 1:
            B[r0 + 1:r1, k].zero_()
            B[k, r0 + 1:r1].zero_()

        # Local neighborhood that can be affected by this reflector for a b-banded matrix.
        # Left edge reaches back by b due to symmetry/band coupling; right edge reaches forward by ~2b.
        c0 = max(0, k - b)
        c1 = min(n, k + 2 * b + 2)  # exclusive
        W = c1 - c0
        if W <= 0:
            continue

        # Left update: C = H^T * C, where C = B[r0:r1, c0:c1] (m, W)
        C = B[r0:r1, c0:c1]
        y = y_buf[:W]
        y.zero_()
        grid = (triton.cdiv(m, 256), triton.cdiv(W, 32))
        gemv_t_partial_kernel[grid](
            C, v, y,
            M=m, N=W,
            stride_am=C.stride(0), stride_an=C.stride(1),
            stride_v=v.stride(0), stride_y=y.stride(0),
            BLOCK_M=256, BLOCK_N=32,
            num_warps=4,
        )
        grid2 = (triton.cdiv(m, 128), triton.cdiv(W, 32))
        rank1_update_rect_tau_ptr_kernel[grid2](
            C, v, y, tau,
            M=m, N=W,
            stride_am=C.stride(0), stride_an=C.stride(1),
            stride_v=v.stride(0), stride_y=y.stride(0),
            BM=128, BN=32,
            num_warps=4,
        )

        # Right update: D = D * H, where D = B[c0:c1, r0:r1] (W, m)
        D = B[c0:c1, r0:r1]
        z = z_buf[:W]
        z.zero_()
        grid3 = (W, triton.cdiv(m, block_n))
        matvec_partial_kernel[grid3](
            D, v, z,
            M=W, N=m,
            stride_am=D.stride(0), stride_an=D.stride(1),
            stride_x=v.stride(0), stride_y=z.stride(0),
            BLOCK_N=block_n,
            num_warps=4,
        )
        grid4 = (triton.cdiv(W, 128), triton.cdiv(m, 32))
        rank1_update_rect_tau_ptr_kernel[grid4](
            D, z, v, tau,
            M=W, N=m,
            stride_am=D.stride(0), stride_an=D.stride(1),
            stride_v=z.stride(0), stride_y=v.stride(0),
            BM=128, BN=32,
            num_warps=4,
        )



def two_stage_tridiag(A: torch.Tensor, b: int, *, bc_dense: bool = True, bc_magma: bool = False, bc_magma_persist: bool = False,
                      stage1: str = "dbbr",
                      grsiz: int = 4,
                      fused_hbtypes: bool = True,
                      workers: int | None = None,
                      mod_store: int | None = None,
                      check: bool = False):
    """
    End-to-end 2-stage tridiagonalization (prototype):
      Stage 1: SBR to band (bandwidth b)
      Stage 2: direct tridiagonalization on the banded matrix (placeholder for BC)
    """
    from triton_dsytrd import dsytrd_triton_

    A0 = A.clone() if check else None
    B = A.clone()

    torch.cuda.synchronize()
    t0 = time.time()
    if stage1 == "sbr":
        sbr_to_band_triton_(B, b)
    elif stage1 == "dbbr":
        sbr_to_band_dbbr_triton_(B, b)
    else:
        raise ValueError(f"unknown stage1={stage1}")
    torch.cuda.synchronize()
    t1 = time.time()

    if bc_magma and bc_magma_persist:
        raise ValueError("choose exactly one of bc_magma or bc_magma_persist")
    if bc_magma:
        d, e = bc_band_to_tridiag_magma_like_(B, b, grsiz=grsiz, fused_hbtypes=fused_hbtypes)
    elif bc_magma_persist:
        d, e = bc_band_to_tridiag_magma_persistent_gpu_(B, b, grsiz=grsiz, workers=workers, mod_store=mod_store)
    else:
        if bc_dense:
            bc_band_to_tridiag_triton_(B, b, dense=True)
        else:
            bc_band_to_tridiag_householder_(B, b)
        d = B.diagonal().clone()
        e = B.diagonal(-1).clone()
    torch.cuda.synchronize()
    t2 = time.time()

    if check:
        # Compare eigenvalues of original vs tridiagonal T.
        T = torch.zeros_like(A)
        T.diagonal().copy_(d)
        idx = torch.arange(A.shape[0] - 1, device=A.device)
        T[idx + 1, idx] = e
        T[idx, idx + 1] = e
        w0 = torch.linalg.eigvalsh(A0)
        w1 = torch.linalg.eigvalsh(T)
        max_abs = (w0 - w1).abs().max().item()
        rel = max_abs / w0.abs().max().item()
    else:
        max_abs = rel = None

    return d, e, (t1 - t0), (t2 - t1), max_abs, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--b", type=int, default=64)
    ap.add_argument("--stage1-update", type=str, default="syr2k", choices=["syr2k", "gemm2", "gemm1"],
                    help="Stage1 trailing update: syr2k (triangular + symmetrize), gemm2 (two dense GEMMs), or gemm1 (one GEMM with K=2b)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--bc-band", action="store_true", help="use band-limited BC (assumes SBR output is truly banded)")
    ap.add_argument("--bc-magma", action="store_true", help="use MAGMA-like hb2st bulge chasing (serial schedule; wantz=0)")
    ap.add_argument("--bc-magma-persist", action="store_true", help="use MAGMA-like hb2st, but GPU-resident scheduling (persistent kernel)")
    ap.add_argument("--bc-pipeline", action="store_true", help="use GPU sweep-pipeline bulge chasing (gCom-style)")
    ap.add_argument("--grsiz", type=int, default=4, help="bulge-chasing group size (MAGMA grsiz)")
    ap.add_argument("--no-fused-hbtypes", action="store_true", help="disable fused hbtype1/2/3 kernels (debug)")
    ap.add_argument("--stage1", type=str, default="dbbr", choices=["dbbr", "sbr"],
                    help="Stage1 method: dbbr (blocked) or sbr")
    ap.add_argument("--workers", type=int, default=0, help="number of GPU worker programs (0=auto). Used by --bc-magma-persist and --bc-pipeline")
    ap.add_argument("--mod-store", type=int, default=0, help="(persistent) ring-buffer depth for V/TAU (0=auto)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    def init_sym_inplace_(A: torch.Tensor, seed: int):
        # Deterministic-ish init without creating extra full-size temporaries.
        g = torch.Generator(device="cuda")
        g.manual_seed(int(seed))
        A.normal_(generator=g)
        # Make symmetric by copying upper from lower (keeps lower as the source of truth).
        grid = (triton.cdiv(A.shape[0], 128), triton.cdiv(A.shape[0], 128))
        symmetrize_upper_from_lower_kernel[grid](
            A, A.shape[0],
            A.stride(0), A.stride(1),
            off=0,
            BM=128, BN=128,
            num_warps=4,
        )
        # Mild diagonal dominance helps avoid pathological cases in debug/checks.
        A.diagonal().add_(float(A.shape[0]))

    if args.bc_magma_persist and args.bc_magma:
        raise SystemExit("error: choose only one of --bc-magma or --bc-magma-persist")
    if args.bc_pipeline and (args.bc_magma or args.bc_magma_persist):
        raise SystemExit("error: --bc-pipeline is mutually exclusive with --bc-magma/--bc-magma-persist")

    # ---- warm once + run once (only the selected path) ----
    workers = None if args.workers == 0 else args.workers
    mod_store = None if args.mod_store == 0 else args.mod_store
    bc_dense = not args.bc_band

    def stage1_inplace(B):
        if args.stage1 == "sbr":
            sbr_to_band_triton_(B, args.b)
        else:
            if args.stage1_update == "syr2k":
                sbr_to_band_dbbr_triton_(B, args.b, update="syr2k", symmetrize=True)
            elif args.stage1_update == "gemm2":
                sbr_to_band_dbbr_triton_(B, args.b, update="gemm2", symmetrize=False)
            else:
                sbr_to_band_dbbr_triton_(B, args.b, update="gemm1", symmetrize=False)

    # Allocate once; do warm+run in-place to avoid OOM at large n.
    A = torch.empty((args.n, args.n), device="cuda", dtype=torch.float64)

    # warm (not timed)
    init_sym_inplace_(A, args.seed)
    stage1_inplace(A)
    if args.bc_pipeline:
        d, e = bc_band_to_tridiag_pipeline_gpu_(A, args.b, workers=workers)
    elif args.bc_magma_persist:
        d, e = bc_band_to_tridiag_magma_persistent_gpu_(A, args.b, grsiz=args.grsiz, workers=workers, mod_store=mod_store)
    else:
        _ = two_stage_tridiag(
            A, args.b,
            bc_dense=bc_dense, bc_magma=args.bc_magma, bc_magma_persist=args.bc_magma_persist,
            stage1=args.stage1,
            grsiz=args.grsiz,
            fused_hbtypes=(not args.no_fused_hbtypes),
            workers=workers,
            mod_store=mod_store,
            check=False,
        )
    # Drop warm outputs before the timed run.
    try:
        del d, e
    except NameError:
        pass
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # run
    init_sym_inplace_(A, args.seed + 1 if args.check else args.seed)
    A0 = A.clone() if args.check else None
    B = A  # operate in-place to reduce peak memory
    torch.cuda.synchronize()
    t0 = time.time()
    stage1_inplace(B)
    torch.cuda.synchronize()
    t1 = time.time()
    if args.bc_pipeline:
        d, e = bc_band_to_tridiag_pipeline_gpu_(B, args.b, workers=workers)
    elif args.bc_magma_persist:
        d, e = bc_band_to_tridiag_magma_persistent_gpu_(B, args.b, grsiz=args.grsiz, workers=workers, mod_store=mod_store)
    else:
        # Fall back to existing driver (supports --bc-magma and dense/band baselines).
        d, e, _, t_stage2, max_abs, rel = two_stage_tridiag(
            A, args.b,
            bc_dense=bc_dense, bc_magma=args.bc_magma, bc_magma_persist=args.bc_magma_persist,
            stage1=args.stage1,
            grsiz=args.grsiz,
            fused_hbtypes=(not args.no_fused_hbtypes),
            workers=workers,
            mod_store=mod_store,
            check=args.check,
        )
        # two_stage_tridiag already includes stage1 timing; override to keep reporting consistent below.
        torch.cuda.synchronize()
        t2 = time.time()
        t_sbr = t1 - t0
        if args.check:
            max_abs, rel = max_abs, rel
        print(f"2stage n={args.n} b={args.b} sbr_ms={t_sbr*1e3:.3f} stage2_ms={t_stage2*1e3:.3f} total_ms={(t_sbr+t_stage2)*1e3:.3f}")
        if args.check:
            print(f"eigval max_abs={max_abs:.3e} rel={rel:.3e}")
        return
    torch.cuda.synchronize()
    t2 = time.time()
    t_sbr = t1 - t0
    t_stage2 = t2 - t1

    if args.check:
        T = torch.zeros_like(A)
        T.diagonal().copy_(d)
        idx = torch.arange(A.shape[0] - 1, device=A.device)
        T[idx + 1, idx] = e
        T[idx, idx + 1] = e
        w0 = torch.linalg.eigvalsh(A0)
        w1 = torch.linalg.eigvalsh(T)
        max_abs = (w0 - w1).abs().max().item()
        rel = max_abs / w0.abs().max().item()
    else:
        max_abs = rel = None
    print(f"2stage n={args.n} b={args.b} sbr_ms={t_sbr*1e3:.3f} stage2_ms={t_stage2*1e3:.3f} total_ms={(t_sbr+t_stage2)*1e3:.3f}")
    if args.check:
        print(f"eigval max_abs={max_abs:.3e} rel={rel:.3e}")


if __name__ == '__main__':
    main()
