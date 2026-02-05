#!/usr/bin/env python3
"""
Best-known Triton-CGY path (as of 2026-02-05) without touching the original CLI code.

Implements: warm once + run once
  - Stage1: sbr_to_band_dbbr_triton_(update="gemm1")
  - Stage2: bc_band_to_tridiag_pipeline_gpu_

Usage:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 triton_dsytrd_run.py --n 45056 --b 64
"""
from __future__ import annotations

import argparse
import time

import torch

import triton
import triton.language as tl


def eff_tflops_sytrd(n: int, ms: float) -> float:
    flops = (4.0 / 3.0) * (float(n) ** 3)
    return flops / (ms * 1e-3) / 1e12


# -----------------------------------------------------------------------------
# Inlined Triton kernels used by the current Triton DSYTRD path (Triton-CGY).
#
# Rationale: keep this file self-contained for the performance-critical kernels,
# without modifying or trimming the original research/prototype code in
# `triton_tridiag_2stage.py`.
#
# Source of truth / provenance: copied from `triton_tridiag_2stage.py` as of
# 2026-02-05 (A100, FP64, uplo=L, b<=64), and used here directly.
# -----------------------------------------------------------------------------

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
    BM: tl.constexpr = 16
    BN: tl.constexpr = 16
    BK: tl.constexpr = 16

    # w = tau * A * v
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

            ii2 = ii[:, None]
            jj2 = jj[None, :]
            mask = mask_i[:, None] & mask_j[None, :] & (ii2 >= jj2)
            diff2 = (ii2 - jj2).to(tl.int32)
            Ablk = tl.load(A_ptr + diff2 * stride0 + (st + jj2) * stride1, mask=mask, other=0.0).to(tl.float64)
            Ablk = Ablk - v_i[:, None] * w_j[None, :] - w_i[:, None] * v_j[None, :]
            tl.store(A_ptr + diff2 * stride0 + (st + jj2) * stride1, Ablk, mask=mask)


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

    tautop = tl.load(TAU_ptr + taupos).to(tl.float64)

    # Right update on C = A(J1:J2, st:ed), shape (lem, len_)
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
    diff0 = (J1 - st) + offs_l
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

    # Left update on A(J1:J2, st+1:ed), shape (lem, len_-1)
    len2 = len_ - 1
    if len2 <= 0:
        return

    for j0 in tl.static_range(0, BLK, BN):
        jj = j0 + tl.arange(0, BN)
        mask_j = jj < len2
        y2 = tl.zeros([BN], dtype=tl.float64)

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
def bc_pipeline_hbtypes_queue_kernel(A_ptr, V_ptr, TAU_ptr, W_ptr, gCom_ptr,
                                     next_sweep_ptr,
                                     n, nb,
                                     stride0, stride1,
                                     MOD_STORE: tl.constexpr,
                                     BLK: tl.constexpr):
    """
    Work-queue variant of the paper-style sweep pipeline.

    Each program repeatedly grabs the next sweep index from a global counter, which keeps the set of active sweeps
    roughly contiguous. This dramatically reduces pathological spinning compared to a fixed strided assignment,
    and avoids deadlock scenarios where a sweep waits on a predecessor sweep that hasn't started yet.
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


def _band_fill_from_dense_lower_triton_(A_band: torch.Tensor, B: torch.Tensor, b: int):
    """Fill MAGMA-like lower band storage A_band (2*b+1,n) from dense B (n,n), using only the lower b-band."""
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


def bc_band_to_tridiag_pipeline_gpu_(B: torch.Tensor, b: int, *, workers: int | None = None):
    """
    Stage 2 (BC): GPU-based bulge chasing (queue-based gCom sweep pipeline) on a banded symmetric matrix.

    This variant is tuned for the Triton DSYTRD defaults: uplo=L, FP64, b<=64.
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
        workers = int(props.multi_processor_count)
    workers = max(1, min(int(workers), max(1, n - 2)))

    V = torch.zeros((workers * n,), device=B.device, dtype=B.dtype)
    TAU = torch.zeros((workers * n,), device=B.device, dtype=B.dtype)
    W = torch.empty((workers * n,), device=B.device, dtype=B.dtype)
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


def _init_sym_inplace_(A: torch.Tensor, seed: int, symm_kernel):
    # Deterministic-ish init without extra full-size temporaries.
    g = torch.Generator(device="cuda")
    g.manual_seed(int(seed))
    A.normal_(generator=g)
    n = A.shape[0]
    grid = (triton.cdiv(n, 128), triton.cdiv(n, 128))
    symm_kernel[grid](
        A, n,
        A.stride(0), A.stride(1),
        off=0,
        BM=128, BN=128,
        num_warps=4,
    )
    A.diagonal().add_(float(n))


def _time_cuda_ms(fn) -> float:
    # warm once + timed run once
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    fn()
    e.record()
    e.synchronize()
    return s.elapsed_time(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--b", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0, help="0 => auto (SM count)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    torch.set_default_dtype(torch.float64)

    import triton_tridiag_2stage as t2

    if args.b > 64:
        raise SystemExit("bc-pipeline path currently assumes b<=64")

    # Allocate once; reuse for warm+run to minimize peak memory.
    A = torch.empty((args.n, args.n), device="cuda", dtype=torch.float64)

    workers = None if args.workers == 0 else args.workers

    def run_once(seed: int):
        _init_sym_inplace_(A, seed, symmetrize_upper_from_lower_kernel)
        torch.cuda.synchronize()

        t0 = time.time()
        t2.sbr_to_band_dbbr_triton_(A, args.b, update="gemm1", symmetrize=False)
        torch.cuda.synchronize()
        t1 = time.time()
        _ = bc_band_to_tridiag_pipeline_gpu_(A, args.b, workers=workers)
        torch.cuda.synchronize()
        t2_end = time.time()
        return (t1 - t0) * 1e3, (t2_end - t1) * 1e3, (t2_end - t0) * 1e3

    # Warm (not reported)
    _ = run_once(args.seed)
    torch.cuda.empty_cache()

    # Run (reported)
    s1_ms, s2_ms, total_ms = run_once(args.seed + 1)
    tf = eff_tflops_sytrd(args.n, total_ms)
    print(f"2stage(triton-dsytrd) n={args.n} b={args.b} sbr_ms={s1_ms:.3f} stage2_ms={s2_ms:.3f} total_ms={total_ms:.3f}")
    print(f"effective_TFLOP/s={tf:.3f} (model=4/3*n^3)")


if __name__ == "__main__":
    main()
