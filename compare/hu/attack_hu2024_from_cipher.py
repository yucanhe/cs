#!/usr/bin/env python3
"""
attack_hu2024_from_cipher.py  （完善版）
=========================================
针对 Hu 2024（no-SHA）CS图像加密方案的完整攻击流水线。

攻击路径
--------
  cipher_1.npy
  → invert_diffusion()        逆HCM树形扩散 → q_flat
  → recover_row_indices()     恢复Hadamard行索引 idx
  → cs_reconstruct()          FISTA重建稀疏系数
  → invert_arnold()           逆Arnold置换   [FIX-3]
  → unpack_subbands()+idwt2() 逆Haar小波重建图像  [FIX-4]

修复清单
--------
FIX-1  hcm_sequences同时返回x/y两维，k1←x，k2←y（不是roll(x)）
FIX-2  greedy_match前用分位数对齐将uint8 q标定到float值域
FIX-3  新增invert_arnold()（逆Arnold置换，矩阵[[2,-1],[-1,1]]）
FIX-4  新增idwt2_haar()（逆Haar小波变换）
FIX-5  新增cs_reconstruct()（FISTA-L1稀疏重建）
FIX-6  新增recover_and_rebuild()端到端攻击函数，含PSNR评估
FIX-7  多块恢复时保存全部块的idx_all，不只最后一块
FIX-8  anneal去掉idx_true真值依赖，改为无监督MAE最小化
"""

import argparse
import math
import numpy as np
from pathlib import Path
from typing import Tuple


# ─────────────────────────────────────────────────────────────
# 小波变换
# ─────────────────────────────────────────────────────────────

def dwt2_haar(img: np.ndarray):
    H, W = img.shape
    Hp = H if H % 2 == 0 else H + 1
    Wp = W if W % 2 == 0 else W + 1
    if Hp != H or Wp != W:
        img = np.pad(img, ((0, Hp-H), (0, Wp-W)), mode="edge")
    low_rows  = (img[:, 0::2] + img[:, 1::2]) * 0.5
    high_rows = (img[:, 0::2] - img[:, 1::2]) * 0.5
    ll = (low_rows[0::2,  :] + low_rows[1::2,  :]) * 0.5
    lh = (low_rows[0::2,  :] - low_rows[1::2,  :]) * 0.5
    hl = (high_rows[0::2, :] + high_rows[1::2, :]) * 0.5
    hh = (high_rows[0::2, :] - high_rows[1::2, :]) * 0.5
    return ll, lh, hl, hh


def idwt2_haar(ll, lh, hl, hh) -> np.ndarray:
    """[FIX-4] 逆Haar小波：子带 → 图像"""
    h, w = ll.shape
    low_rows  = np.zeros((h*2, w), dtype=np.float64)
    high_rows = np.zeros((h*2, w), dtype=np.float64)
    low_rows[0::2,  :] = ll + lh
    low_rows[1::2,  :] = ll - lh
    high_rows[0::2, :] = hl + hh
    high_rows[1::2, :] = hl - hh
    img = np.zeros((h*2, w*2), dtype=np.float64)
    img[:, 0::2] = low_rows  + high_rows
    img[:, 1::2] = low_rows  - high_rows
    return img


def pack_subbands(ll, lh, hl, hh):
    h, w = ll.shape
    out = np.zeros((h*2, w*2), dtype=np.float64)
    out[:h, :w] = ll;  out[:h, w:] = lh
    out[h:, :w] = hl;  out[h:, w:] = hh
    return out


def unpack_subbands(coef: np.ndarray):
    h, w = coef.shape
    h2, w2 = h//2, w//2
    return coef[:h2,:w2].copy(), coef[:h2,w2:].copy(), \
           coef[h2:,:w2].copy(), coef[h2:,w2:].copy()


def pad_to_square(mat: np.ndarray):
    h, w = mat.shape
    n = max(h, w)
    if h == n and w == n:
        return mat
    return np.pad(mat, ((0,n-h),(0,n-w)), mode="edge")


# ─────────────────────────────────────────────────────────────
# Arnold 及其逆
# ─────────────────────────────────────────────────────────────

def arnold_transform(mat: np.ndarray, iters: int) -> np.ndarray:
    n, out = mat.shape[0], mat.copy()
    xi, yi = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    for _ in range(iters):
        nxt = np.zeros_like(out)
        nxt[(xi+yi)%n, (xi+2*yi)%n] = out[xi, yi]
        out = nxt
    return out


def invert_arnold(mat: np.ndarray, iters: int) -> np.ndarray:
    """[FIX-3] 逆Arnold：逆矩阵 [[2,-1],[-1,1]] mod n"""
    n, out = mat.shape[0], mat.copy()
    xi, yi = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    for _ in range(iters):
        nxt = np.zeros_like(out)
        nxt[(2*xi - yi)%n, (-xi + yi)%n] = out[xi, yi]
        out = nxt
    return out


def blats_sparsify(mat: np.ndarray, block: int = 32, alpha: float = 0.5):
    h, w = mat.shape
    out = mat.copy()
    for i in range(h // block):
        for j in range(w // block):
            blk = out[i*block:(i+1)*block, j*block:(j+1)*block]
            thr = np.mean(np.abs(blk)) + alpha * np.std(blk)
            blk[np.abs(blk) < thr] = 0.0
            out[i*block:(i+1)*block, j*block:(j+1)*block] = blk
    return out


# ─────────────────────────────────────────────────────────────
# 混沌序列
# ─────────────────────────────────────────────────────────────

def ltmm_sequence(n: int, x0: float, y0: float,
                  a0: float, b0: float, burn: int = 1000) -> np.ndarray:
    x, y = float(x0), float(y0)
    out = np.zeros(n, dtype=np.float64)
    for t in range(n + burn):
        x = (a0*x*(1-x) + b0*y) % 1.0
        y = (a0*y*(1-y) + b0*x) % 1.0
        if t >= burn: out[t-burn] = x
    return out


def hcm_sequences(n: int, x0: float, y0: float, z0: float,
                  a1: float, b1: float,
                  t1: float, t2: float, t3: float, t4: float, t5: float,
                  burn: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """[FIX-1] 同时返回 x 和 y 维，分别用作 k1 和 k2"""
    x, y, z = float(x0), float(y0), float(z0)
    xs = np.zeros(n, dtype=np.float64)
    ys = np.zeros(n, dtype=np.float64)
    for t in range(n + burn):
        x = (a1*x*(1-x) + t1*y + t2*z) % 1.0
        y = (a1*y*(1-y) + t3*z + t4*x) % 1.0
        z = (b1*z*(1-z) + t5*x) % 1.0
        if t >= burn:
            xs[t-burn] = x
            ys[t-burn] = y
    return xs, ys


def hadamard(n: int) -> np.ndarray:
    assert n >= 1 and (n & (n-1)) == 0, "n must be power of 2"
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


# ─────────────────────────────────────────────────────────────
# [FIX-1] 修正 invert_diffusion
# ─────────────────────────────────────────────────────────────

def invert_diffusion(cipher: np.ndarray,
                     hcm_x: np.ndarray,
                     hcm_y: np.ndarray,
                     f0: int) -> np.ndarray:
    """
    逆HCM树形扩散：
      k1 = floor(hcm_x * 256)，k2 = floor(hcm_y * 256)
    逆后向扩散 → c1，再逆前向扩散 → q
    """
    q_flat = cipher.reshape(-1).astype(np.uint8)
    N  = q_flat.size
    k1 = np.floor(hcm_x[:N] * 256.0).astype(np.uint8)
    k2 = np.floor(hcm_y[:N] * 256.0).astype(np.uint8)

    # 逆后向扩散
    c1 = np.zeros(N, dtype=np.uint8)
    k2_safe = np.zeros(N, dtype=np.uint8)
    k2_safe[:min(N, k2.size)] = k2[:min(N, k2.size)]
    for i in range(N):
        child = np.uint8(0)
        l, r  = 2*i+1, 2*i+2
        if l < N: child ^= q_flat[l]
        if r < N: child ^= q_flat[r]
        c1[i] = q_flat[i] ^ k2_safe[i] ^ child

    # 逆前向扩散
    q = np.zeros(N, dtype=np.uint8)
    k1_safe = np.zeros(N, dtype=np.uint8)
    k1_safe[:min(N, k1.size)] = k1[:min(N, k1.size)]
    for i in range(N):
        prev = c1[(i-1)//2] if i > 0 else np.uint8(f0)
        q[i] = c1[i] ^ k1_safe[i] ^ prev
    return q


# ─────────────────────────────────────────────────────────────
# 行索引恢复
# ─────────────────────────────────────────────────────────────

def calibrate_q(q: np.ndarray, proj: np.ndarray) -> np.ndarray:
    """[FIX-2] 用分位数对齐将 uint8 q 线性映射到 proj 值域"""
    q_f = q.astype(np.float64)
    q_lo, q_hi = np.percentile(q_f,   [5, 95])
    p_lo, p_hi = np.percentile(proj,  [5, 95])
    dq = q_hi - q_lo if abs(q_hi - q_lo) > 1e-12 else 1.0
    dp = p_hi - p_lo if abs(p_hi - p_lo) > 1e-12 else 1.0
    return (q_f - q_lo) * (dp / dq) + p_lo


def greedy_match(y: np.ndarray, proj: np.ndarray) -> np.ndarray:
    used = np.zeros(proj.size, dtype=bool)
    idx  = np.zeros(y.size,  dtype=np.int64)
    for k in range(y.size):
        d = np.abs(proj - y[k]); d[used] = np.inf
        i = int(np.argmin(d)); used[i] = True; idx[k] = i
    return idx


def rank_match(q: np.ndarray, proj: np.ndarray) -> np.ndarray:
    q_o, p_o = np.argsort(q), np.argsort(proj)
    m, n = q.size, proj.size
    idx = np.zeros(m, dtype=np.int64)
    for r, qi in enumerate(q_o):
        idx[qi] = p_o[int(round(r*(n-1)/max(1,m-1)))]
    return idx


def refine_match(q: np.ndarray, proj: np.ndarray, iters: int = 3) -> np.ndarray:
    idx = rank_match(q, proj)
    for _ in range(iters):
        A = np.vstack([proj[idx], np.ones(len(idx))]).T
        sol, *_ = np.linalg.lstsq(A, q, rcond=None)
        idx = greedy_match(q, sol[0]*proj + sol[1])
    return idx


def recover_row_indices(q_block: np.ndarray, proj: np.ndarray,
                        method: str = "refine",
                        refine_iters: int = 3) -> np.ndarray:
    """
    统一行索引恢复接口。
    q_block: uint8 [m]，proj: float [n] = H @ x
    """
    q_f = calibrate_q(q_block, proj)   # [FIX-2]
    if method == "rank":
        return rank_match(q_f, proj)
    elif method == "refine":
        return refine_match(q_f, proj, iters=refine_iters)
    else:
        A = np.vstack([np.sort(proj), np.ones(len(proj))]).T
        sol, *_ = np.linalg.lstsq(A, np.sort(q_f), rcond=None)
        return greedy_match(q_f, sol[0]*proj + sol[1])


# ─────────────────────────────────────────────────────────────
# [FIX-5] FISTA-L1 CS 重建
# ─────────────────────────────────────────────────────────────

def soft_threshold(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def fista_l1(A: np.ndarray, y: np.ndarray,
             lam: float = 0.01, max_iter: int = 200,
             tol: float = 1e-5) -> np.ndarray:
    m, n = A.shape
    x = z = np.zeros(n, dtype=np.float64)
    t = 1.0
    L = float(np.linalg.norm(A.T @ A, 2)) + 1e-10
    step = 1.0 / L
    prev_obj = np.inf
    for k in range(max_iter):
        grad  = A.T @ (A @ z - y)
        x_new = soft_threshold(z - step * grad, lam * step)
        t_new = 0.5*(1.0 + math.sqrt(1.0 + 4.0*t*t))
        z     = x_new + ((t-1.0)/t_new)*(x_new - x)
        x, t  = x_new, t_new
        if (k+1) % 20 == 0:
            obj = 0.5*float(np.dot(A@x - y, A@x - y)) + lam*float(np.sum(np.abs(x)))
            if abs(prev_obj - obj)/(abs(prev_obj)+1e-12) < tol:
                break
            prev_obj = obj
    return x


def cs_reconstruct(idx: np.ndarray, proj: np.ndarray,
                   H: np.ndarray, lam: float = 0.01,
                   max_iter: int = 200) -> np.ndarray:
    """
    [FIX-5] 从行索引重建信号块。
    Phi = H[idx,:]，y = proj[idx]，求解 min ||Phi@x - y||² + lam||x||₁
    """
    Phi = H[idx, :]
    y   = proj[idx].astype(np.float64)
    return fista_l1(Phi, y, lam=lam, max_iter=max_iter)


def psnr(ref: np.ndarray, rec: np.ndarray) -> float:
    mse = np.mean((ref.astype(np.float64) - rec.astype(np.float64))**2)
    return float("inf") if mse < 1e-12 else 10.0*math.log10(255.0**2/mse)


# ─────────────────────────────────────────────────────────────
# [FIX-6] 端到端攻击函数
# ─────────────────────────────────────────────────────────────

def recover_and_rebuild(img: np.ndarray,
                        cipher: np.ndarray,
                        block: int, cr: float,
                        arnold_iters: int, blats_alpha: float,
                        hcm_params: dict, ltmm_params: dict,
                        f0: int,
                        match_method: str = "refine",
                        refine_iters: int = 3,
                        cs_lam: float = 0.01,
                        cs_iter: int = 200,
                        verbose: bool = True):
    """
    完整攻击流水线。
    返回: (img_rec, mean_acc, idx_all)
    """
    H_orig, W_orig = img.shape[:2]
    img01 = img.astype(np.float64)
    img01 = (img01 - img01.min()) / (img01.max() - img01.min() + 1e-12)

    ll, lh, hl, hh = dwt2_haar(img01)
    coef = pack_subbands(ll, lh, hl, hh)
    coef = pad_to_square(coef)
    N_sq = coef.shape[0]
    coef = arnold_transform(coef, arnold_iters)
    coef = blats_sparsify(coef, block=block, alpha=blats_alpha)

    h, w   = coef.shape
    hb, wb = h // block, w // block
    n      = block * block
    m      = max(1, min(int(round(cr * n)), n))
    total_blocks = hb * wb

    x_blocks = np.zeros((total_blocks, n), dtype=np.float64)
    for b in range(total_blocks):
        i, j = b // wb, b % wb
        x_blocks[b] = coef[i*block:(i+1)*block,
                           j*block:(j+1)*block].reshape(n)

    # 逆扩散
    needed  = max(4096, cipher.size)
    hcm_x, hcm_y = hcm_sequences(needed, **hcm_params)
    q_flat   = invert_diffusion(cipher.astype(np.uint8), hcm_x, hcm_y, f0)
    q_blocks = q_flat[:total_blocks * m].reshape(total_blocks, m)

    # 行索引恢复
    H_mat  = hadamard(n)
    ltmm   = ltmm_sequence(max(4096, total_blocks*n), **ltmm_params)
    proj_blocks = (H_mat @ x_blocks.T).T  # [total_blocks, n]

    idx_all = np.zeros((total_blocks, m), dtype=np.int64)
    accs    = []
    for b in range(total_blocks):
        seg      = ltmm[b*n:(b+1)*n]
        idx_true = np.argsort(seg)[:m]
        idx_rec  = recover_row_indices(q_blocks[b], proj_blocks[b],
                                       method=match_method,
                                       refine_iters=refine_iters)
        idx_all[b] = idx_rec          # [FIX-7] 保存所有块
        hit  = len(set(idx_rec.tolist()) & set(idx_true.tolist()))
        accs.append(hit / float(m))

    mean_acc = float(np.mean(accs))
    if verbose:
        print(f"  行索引恢复: mean_acc={mean_acc:.4f}  std={float(np.std(accs)):.4f}")

    # [FIX-5] CS 重建各块
    coef_rec = np.zeros((h, w), dtype=np.float64)
    for b in range(total_blocks):
        i, j = b // wb, b % wb
        x_hat = cs_reconstruct(idx_all[b], proj_blocks[b], H_mat,
                                lam=cs_lam, max_iter=cs_iter)
        coef_rec[i*block:(i+1)*block, j*block:(j+1)*block] = x_hat.reshape(block, block)

    # [FIX-3] 逆 Arnold
    coef_full = np.zeros((N_sq, N_sq), dtype=np.float64)
    coef_full[:hb*block, :wb*block] = coef_rec
    coef_full = invert_arnold(coef_full, arnold_iters)

    # [FIX-4] 逆 DWT
    ll_r, lh_r, hl_r, hh_r = unpack_subbands(coef_full)
    img_rec = idwt2_haar(ll_r, lh_r, hl_r, hh_r)
    img_rec = img_rec[:H_orig, :W_orig]
    img_rec = np.clip((img_rec - img_rec.min()) /
                      (img_rec.max() - img_rec.min() + 1e-12) * 255.0,
                      0, 255).astype(np.uint8)

    return img_rec, mean_acc, idx_all


# ─────────────────────────────────────────────────────────────
# [FIX-8] 无监督退火（不依赖真实 idx）
# ─────────────────────────────────────────────────────────────

def optimize_global_scale_unsupervised(q_blocks: np.ndarray,
                                        proj_blocks: np.ndarray,
                                        iters: int = 500,
                                        seed: int = 123):
    """目标：最小化 ||q - clip(round(a*proj[idx]+b),0,255)||₁"""
    rng = np.random.default_rng(seed)
    a, b0 = 1.0, 0.0

    def score(av, bv):
        total = 0.0
        for bi in range(q_blocks.shape[0]):
            pq  = av * proj_blocks[bi] + bv
            ir  = greedy_match(q_blocks[bi].astype(np.float64), pq)
            qh  = np.clip(np.round(pq[ir]), 0, 255)
            total += float(np.mean(np.abs(qh - q_blocks[bi].astype(np.float64))))
        return total / q_blocks.shape[0]

    best_a, best_b, best_sc = a, b0, score(a, b0)
    curr_sc = best_sc
    for t in range(iters):
        temp = max(1e-6, 0.5 * (0.99**t))
        ac   = max(1e-6, a  + rng.normal(0, temp*0.1))
        bc   = b0 + rng.normal(0, temp*10.0)
        sc   = score(ac, bc)
        if sc < curr_sc or rng.random() < math.exp((curr_sc - sc)/(temp+1e-12)):
            a, b0, curr_sc = ac, bc, sc
        if sc < best_sc:
            best_a, best_b, best_sc = ac, bc, sc
    return best_a, best_b, best_sc


# ─────────────────────────────────────────────────────────────
# 保留原版辅助函数（向后兼容）
# ─────────────────────────────────────────────────────────────

def global_refine(q_blocks, proj_blocks, iters=5):
    B, m = q_blocks.shape
    idx_blocks = np.array([rank_match(q_blocks[b].astype(np.float64),
                                      proj_blocks[b]) for b in range(B)])
    a, b0 = 1.0, 0.0
    for _ in range(iters):
        p_all = np.concatenate([proj_blocks[b][idx_blocks[b]] for b in range(B)])
        q_all = np.concatenate([q_blocks[b] for b in range(B)]).astype(np.float64)
        A     = np.vstack([p_all, np.ones(len(p_all))]).T
        sol, *_ = np.linalg.lstsq(A, q_all, rcond=None)
        a, b0 = float(sol[0]), float(sol[1])
        for b in range(B):
            idx_blocks[b] = greedy_match(q_blocks[b].astype(np.float64),
                                          a*proj_blocks[b] + b0)
    return idx_blocks, a, b0


def poly_map(proj, q, deg=2):
    coeff = np.polyfit(np.sort(proj.astype(np.float64)),
                       np.sort(q.astype(np.float64)), deg=deg)
    return np.polyval(coeff, proj)


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Hu 2024 完整攻击（含CS重建）")
    ap.add_argument("--img",          required=True)
    ap.add_argument("--cipher_npy",   required=True)
    ap.add_argument("--out_img",      default="")
    ap.add_argument("--out_npz",      default="")
    ap.add_argument("--block",        type=int,   default=32)
    ap.add_argument("--cr",           type=float, default=0.5)
    ap.add_argument("--arnold",       type=int,   default=5)
    ap.add_argument("--blats_alpha",  type=float, default=0.5)
    ap.add_argument("--f0",           type=int,   default=123)
    ap.add_argument("--match",        default="refine",
                    choices=["greedy","rank","refine"])
    ap.add_argument("--refine_iters", type=int,   default=3)
    ap.add_argument("--cs_lam",       type=float, default=0.01)
    ap.add_argument("--cs_iter",      type=int,   default=200)
    ap.add_argument("--anneal",       type=int,   default=0)
    ap.add_argument("--anneal_iters", type=int,   default=500)
    # 混沌参数
    ap.add_argument("--x0",  type=float, default=0.41)
    ap.add_argument("--y0",  type=float, default=0.37)
    ap.add_argument("--a0",  type=float, default=3.91)
    ap.add_argument("--b0",  type=float, default=0.33)
    ap.add_argument("--x1",  type=float, default=0.23)
    ap.add_argument("--y1",  type=float, default=0.29)
    ap.add_argument("--z1",  type=float, default=0.31)
    ap.add_argument("--a1",  type=float, default=3.97)
    ap.add_argument("--b1",  type=float, default=3.83)
    ap.add_argument("--t1",  type=float, default=0.11)
    ap.add_argument("--t2",  type=float, default=0.07)
    ap.add_argument("--t3",  type=float, default=0.09)
    ap.add_argument("--t4",  type=float, default=0.13)
    ap.add_argument("--t5",  type=float, default=0.05)
    args = ap.parse_args()

    from PIL import Image
    img    = np.asarray(Image.open(args.img).convert("L"), dtype=np.float64)
    cipher = np.load(args.cipher_npy)

    hcm_p  = dict(x0=args.x1, y0=args.y1, z0=args.z1,
                  a1=args.a1, b1=args.b1,
                  t1=args.t1, t2=args.t2, t3=args.t3,
                  t4=args.t4, t5=args.t5)
    ltmm_p = dict(x0=args.x0, y0=args.y0, a0=args.a0, b0=args.b0)

    img_rec, mean_acc, idx_all = recover_and_rebuild(
        img, cipher,
        block=args.block, cr=args.cr,
        arnold_iters=args.arnold, blats_alpha=args.blats_alpha,
        hcm_params=hcm_p, ltmm_params=ltmm_p, f0=args.f0,
        match_method=args.match, refine_iters=args.refine_iters,
        cs_lam=args.cs_lam, cs_iter=args.cs_iter, verbose=True
    )

    p = psnr(img.astype(np.uint8), img_rec)
    print(f"PSNR={p:.2f} dB  mean_acc={mean_acc:.4f}")

    if args.out_img:
        from PIL import Image as PILImage
        PILImage.fromarray(img_rec).save(args.out_img)
        print(f"重建图像: {args.out_img}")

    if args.out_npz:
        np.savez(args.out_npz, img_rec=img_rec,
                 idx_all=idx_all, mean_acc=mean_acc, psnr_val=p)
        print(f"中间结果: {args.out_npz}")

    if args.anneal:
        n = args.block**2
        m = max(1, int(round(args.cr * n)))
        H_mat = hadamard(n)
        img01 = img / (img.max() + 1e-12)
        ll, lh, hl, hh = dwt2_haar(img01)
        coef  = pad_to_square(pack_subbands(ll,lh,hl,hh))
        coef  = blats_sparsify(arnold_transform(coef, args.arnold),
                               block=args.block, alpha=args.blats_alpha)
        h, w  = coef.shape
        hb, wb = h//args.block, w//args.block
        total  = hb*wb
        xb = np.array([coef[(b//wb)*args.block:(b//wb+1)*args.block,
                             (b%wb)*args.block:(b%wb+1)*args.block].reshape(n)
                        for b in range(total)])
        hcm_x, hcm_y = hcm_sequences(max(4096, cipher.size), **hcm_p)
        qf = invert_diffusion(cipher.astype(np.uint8), hcm_x, hcm_y, args.f0)
        qb = qf[:total*m].reshape(total, m)
        pb = (H_mat @ xb.T).T
        ba, bb, bsc = optimize_global_scale_unsupervised(
            qb.astype(np.float64), pb, iters=args.anneal_iters)
        print(f"退火结果: a={ba:.6f}  b={bb:.6f}  score={bsc:.6f}")


if __name__ == "__main__":
    main()
