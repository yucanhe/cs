#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo3.py (UPDATED)
Full pipeline (invertible cipher_rgba.png) + full measurement module
Reconstruction solver: FIXED to FISTA (no OMP)

Adds (for robustness / attacked cipher tests):
- --cipher_in PATH        : decrypt from an external cipher_rgba.png
- --decrypt_only 1        : only decrypt (requires state file)
- --encrypt_only 1        : only encrypt (skip decrypt + metrics)
- state saved to: out/state.npz
- --state_in PATH         : provide state.npz explicitly (for decrypt_only)
  (If omitted in decrypt_only, will try:
     1) <dir_of_cipher_in>/state.npz
     2) <out>/state.npz )

Keeps:
- Dynamic sensing matrix Phi_i per block (can force static via --static_phi 1)
- 3D float diffusion (can disable via ablate)
- Quantize to uint32
- uint32 ARX diffusion with (dynamic/static/no) S-box (ablate)
- Final ciphertext container: cipher_rgba.png (RGBA stores uint32), decrypt starts from it
- Full evaluation:
    * time/throughput
    * MAE/PSNR/SSIM
    * entropy & adjacent correlation on cipher_uint8 view
    * differential (flip 1 plaintext pixel) NPCR/UACI on cipher_uint8 view
    * key sensitivity (flip 1 key bit) NPCR/UACI on cipher_uint8 view
    * wrong-key decryption quality (from cipher_rgba.png)
    * save images + histograms + correlation scatter + report.txt

Notes:
- Uses Pillow, numpy, scipy, matplotlib, scikit-image(optional), numba(optional)
- Designed to be drop-in runnable.
"""

import argparse
import hashlib
import os
import time
import json
import numpy as np
from dataclasses import dataclass
from scipy.fftpack import dct, idct
from PIL import Image
import matplotlib.pyplot as plt

try:
    from skimage.metrics import structural_similarity as ssim_skimage
except Exception:
    ssim_skimage = None

# -----------------------
# Optional numba
# -----------------------
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def deco(fn): return fn
        return deco

UINT32_MOD = 2**32
MOD_MASK = 0xFFFFFFFF


# ============================================================
# Basic utils
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def frac_arr(x: np.ndarray) -> np.ndarray:
    return x - np.floor(x)

def dct2(a: np.ndarray) -> np.ndarray:
    return dct(dct(a.T, norm="ortho").T, norm="ortho")

def idct2(a: np.ndarray) -> np.ndarray:
    return idct(idct(a.T, norm="ortho").T, norm="ortho")

def blockify(img: np.ndarray, B: int = 8):
    H, W = img.shape
    assert H % B == 0 and W % B == 0, "H,W must be divisible by B"
    blocks = []
    for r in range(0, H, B):
        for c in range(0, W, B):
            blocks.append(img[r:r+B, c:c+B])
    return np.array(blocks), (H // B, W // B)

def unblockify(blocks: np.ndarray, grid_shape, B: int = 8):
    gh, gw = grid_shape
    H, W = gh * B, gw * B
    out = np.zeros((H, W), dtype=blocks.dtype)
    idx = 0
    for r in range(gh):
        for c in range(gw):
            out[r*B:(r+1)*B, c*B:(c+1)*B] = blocks[idx]
            idx += 1
    return out

def _hash_to_seed(key: str) -> int:
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)

def flip_one_bit_in_key(key: str) -> str:
    b = bytearray(key.encode("utf-8"))
    if len(b) == 0:
        b = bytearray(b"\x00")
    b[0] ^= 0x01
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.hex()

def clamp01(x):
    return np.clip(x, 0.0, 1.0)

def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)

def safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return int(default)


# ============================================================
# Python-safe rotations (CRITICAL for non-numba correctness)
# ============================================================

def rotl32_py(x: int, r: int) -> int:
    r &= 31
    x &= MOD_MASK
    return ((x << r) | (x >> (32 - r))) & MOD_MASK

def rotr32_py(x: int, r: int) -> int:
    r &= 31
    x &= MOD_MASK
    return ((x >> r) | (x << (32 - r))) & MOD_MASK


# ============================================================
# Chaos generator (demo SPCMM-like; deterministic)
# ============================================================

def spcmm_generate_xyz(key: str, n: int, burn: int = 2000):
    seed = _hash_to_seed(key)
    rng = np.random.default_rng(seed)

    x = rng.random()
    y = rng.random()
    z = rng.random()

    a = 3.999
    b = 3.985
    c = 3.975
    k1 = 0.12
    k2 = 0.18
    k3 = 0.15

    xs = np.empty(n, dtype=np.float64)
    ys = np.empty(n, dtype=np.float64)
    zs = np.empty(n, dtype=np.float64)

    total = n + burn
    for t in range(total):
        x_next = (a * x * (1 - x) + k1 * np.sin(np.pi * (y + z))) % 1.0
        y_next = (b * y * (1 - y) + k2 * np.sin(np.pi * (x + z))) % 1.0
        z_next = (c * z * (1 - z) + k3 * np.sin(np.pi * (x + y))) % 1.0
        x, y, z = x_next, y_next, z_next
        if t >= burn:
            i = t - burn
            xs[i], ys[i], zs[i] = x, y, z
    return xs, ys, zs


# ============================================================
# Dynamic sensing matrix Phi_i
# ============================================================

def phi_from_key_z(key: str, z_val: float, block_id: int, m_meas: int, n: int = 64):
    msg = f"{key}|{z_val:.16f}|{block_id}|{m_meas}|{n}"
    seed = _hash_to_seed(msg)
    rng = np.random.default_rng(seed)

    A = rng.standard_normal((m_meas, n)).astype(np.float64)

    if m_meas == n:
        Q, R = np.linalg.qr(A)
        s = np.sign(np.diag(R))
        s[s == 0] = 1.0
        Q = Q * s
        return Q.astype(np.float64)

    A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    return A

def phi_static(key: str, m_meas: int, n: int = 64):
    seed = _hash_to_seed(f"{key}|static_phi|{m_meas}|{n}")
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m_meas, n)).astype(np.float64)
    if m_meas == n:
        Q, R = np.linalg.qr(A)
        s = np.sign(np.diag(R))
        s[s == 0] = 1.0
        Q = Q * s
        return Q.astype(np.float64)
    A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    return A


# ============================================================
# Float 3D diffusion (numba optional)
# ============================================================

@njit(cache=True)
def diffusion3d_forward_numba(Yn, mask, a, b, c):
    H, W, M = Yn.shape
    D = np.empty_like(Yn)
    for i in range(H):
        for j in range(W):
            for k in range(M):
                up = D[i-1, j, k] if i > 0 else 0.0
                left = D[i, j-1, k] if j > 0 else 0.0
                prev = D[i, j, k-1] if k > 0 else 0.0
                v = Yn[i, j, k] + a*up + b*left + c*prev + mask[i, j, k]
                D[i, j, k] = v - np.floor(v)
    return D

@njit(cache=True)
def diffusion3d_inverse_numba(D, mask, a, b, c):
    H, W, M = D.shape
    Yn = np.empty_like(D)
    for i in range(H):
        for j in range(W):
            for k in range(M):
                up = D[i-1, j, k] if i > 0 else 0.0
                left = D[i, j-1, k] if j > 0 else 0.0
                prev = D[i, j, k-1] if k > 0 else 0.0
                v = D[i, j, k] - a*up - b*left - c*prev - mask[i, j, k]
                Yn[i, j, k] = v - np.floor(v)
    return Yn

def diffusion3d_forward(Yn, mask, a=0.21, b=0.33, c=0.27, use_numba=True):
    if use_numba and NUMBA_AVAILABLE:
        return diffusion3d_forward_numba(Yn, mask, a, b, c)
    H, W, M = Yn.shape
    D = np.empty_like(Yn, dtype=np.float64)
    for i in range(H):
        for j in range(W):
            for k in range(M):
                up = D[i-1, j, k] if i > 0 else 0.0
                left = D[i, j-1, k] if j > 0 else 0.0
                prev = D[i, j, k-1] if k > 0 else 0.0
                v = Yn[i, j, k] + a*up + b*left + c*prev + mask[i, j, k]
                D[i, j, k] = v - np.floor(v)
    return D

def diffusion3d_inverse(D, mask, a=0.21, b=0.33, c=0.27, use_numba=True):
    if use_numba and NUMBA_AVAILABLE:
        return diffusion3d_inverse_numba(D, mask, a, b, c)
    H, W, M = D.shape
    Yn = np.empty_like(D, dtype=np.float64)
    for i in range(H):
        for j in range(W):
            for k in range(M):
                up = D[i-1, j, k] if i > 0 else 0.0
                left = D[i, j-1, k] if j > 0 else 0.0
                prev = D[i, j, k-1] if k > 0 else 0.0
                v = D[i, j, k] - a*up - b*left - c*prev - mask[i, j, k]
                Yn[i, j, k] = v - np.floor(v)
    return Yn


# ============================================================
# uint32 helpers + dynamic S-box layers
# ============================================================

def u32_from_float01(D: np.ndarray) -> np.ndarray:
    D = frac_arr(D.astype(np.float64))
    return np.floor(D * UINT32_MOD).astype(np.uint32)

def float01_from_u32(U: np.ndarray) -> np.ndarray:
    return (U.astype(np.float64) / UINT32_MOD)

def u32_keystream(key: str, shape) -> np.ndarray:
    seed = _hash_to_seed(key)
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**32, size=np.prod(shape), dtype=np.uint32).reshape(shape)

@njit(cache=True)
def rotl32_numba(x, r):
    r &= 31
    return ((x << r) | (x >> (32 - r))) & MOD_MASK

@njit(cache=True)
def rotr32_numba(x, r):
    r &= 31
    return ((x >> r) | (x << (32 - r))) & MOD_MASK

@njit(cache=True)
def sub_bytes_u32_numba(x, sbox):
    b0 = (x >> 24) & 0xFF
    b1 = (x >> 16) & 0xFF
    b2 = (x >> 8) & 0xFF
    b3 = x & 0xFF
    y0 = int(sbox[b0])
    y1 = int(sbox[b1])
    y2 = int(sbox[b2])
    y3 = int(sbox[b3])
    return ((y0 << 24) | (y1 << 16) | (y2 << 8) | y3) & MOD_MASK

def sub_bytes_u32_py(x: int, sbox: np.ndarray) -> int:
    x &= MOD_MASK
    b0 = (x >> 24) & 0xFF
    b1 = (x >> 16) & 0xFF
    b2 = (x >> 8) & 0xFF
    b3 = x & 0xFF
    y0 = int(sbox[b0])
    y1 = int(sbox[b1])
    y2 = int(sbox[b2])
    y3 = int(sbox[b3])
    return ((y0 << 24) | (y1 << 16) | (y2 << 8) | y3) & MOD_MASK

def build_sbox_layers_from_chaos(key: str, M: int, label: str):
    sbox_layers = np.empty((M, 256), dtype=np.uint8)
    inv_layers = np.empty((M, 256), dtype=np.uint8)
    for kk in range(M):
        xs, _, _ = spcmm_generate_xyz(f"{key}|{label}|depth={kk}", 256 + 64)
        seq = xs[:256]
        perm = np.argsort(seq).astype(np.uint8)
        sbox_layers[kk] = perm
        inv = np.empty(256, dtype=np.uint8)
        inv[sbox_layers[kk]] = np.arange(256, dtype=np.uint8)
        inv_layers[kk] = inv
    return sbox_layers, inv_layers

def build_static_sbox_layers(key: str, M: int, label: str):
    xs, _, _ = spcmm_generate_xyz(f"{key}|{label}|static", 256 + 64)
    perm = np.argsort(xs[:256]).astype(np.uint8)
    inv = np.empty(256, dtype=np.uint8)
    inv[perm] = np.arange(256, dtype=np.uint8)
    sbox_layers = np.repeat(perm[None, :], M, axis=0)
    inv_layers  = np.repeat(inv[None, :],  M, axis=0)
    return sbox_layers, inv_layers

def build_identity_sbox_layers(M: int):
    s = np.arange(256, dtype=np.uint8)
    sbox_layers = np.repeat(s[None, :], M, axis=0)
    inv_layers  = np.repeat(s[None, :], M, axis=0)
    return sbox_layers, inv_layers


# ============================================================
# uint32 ARX + S-box per depth (forward/inverse)
# ============================================================

@njit(cache=True)
def u32_pass1_forward_numba(U, mask1, ks1, sbox_layers, depth_mask):
    H, W, M = U.shape
    C = np.empty_like(U)
    for i in range(H):
        for j in range(W):
            for k in range(M):
                if depth_mask[k] == 0:
                    C[i, j, k] = U[i, j, k]
                    continue
                up = int(C[i-1, j, k]) if i > 0 else 0
                left = int(C[i, j-1, k]) if j > 0 else 0
                prev = int(C[i, j, k-1]) if k > 0 else 0

                s = (int(U[i, j, k])
                     + rotl32_numba(up, 7)
                     + rotl32_numba(left, 11)
                     + rotl32_numba(prev, 19)
                     + int(mask1[i, j, k])) & MOD_MASK

                t = rotl32_numba(s, 3) ^ int(ks1[i, j, k])
                t = sub_bytes_u32_numba(t, sbox_layers[k])
                C[i, j, k] = np.uint32(t)
    return C

@njit(cache=True)
def u32_pass2_backward_numba(C, mask2, ks2, sbox_layers, depth_mask):
    H, W, M = C.shape
    for i in range(H - 1, -1, -1):
        for j in range(W - 1, -1, -1):
            for k in range(M - 1, -1, -1):
                if depth_mask[k] == 0:
                    continue
                dn = int(C[i+1, j, k]) if i + 1 < H else 0
                rt = int(C[i, j+1, k]) if j + 1 < W else 0
                nxt = int(C[i, j, k+1]) if k + 1 < M else 0

                s = (int(C[i, j, k])
                     + rotl32_numba(dn, 5)
                     + rotl32_numba(rt, 13)
                     + rotl32_numba(nxt, 17)
                     + int(mask2[i, j, k])) & MOD_MASK

                t = rotl32_numba(s, 9) ^ int(ks2[i, j, k])
                t = sub_bytes_u32_numba(t, sbox_layers[k])
                C[i, j, k] = np.uint32(t)
    return C

@njit(cache=True)
def u32_undo_pass2_forward_numba(C, mask2, ks2, inv_sbox_layers, depth_mask):
    H, W, M = C.shape
    C1 = C.copy()
    for i in range(H):
        for j in range(W):
            for k in range(M):
                if depth_mask[k] == 0:
                    continue
                dn = int(C1[i+1, j, k]) if i + 1 < H else 0
                rt = int(C1[i, j+1, k]) if j + 1 < W else 0
                nxt = int(C1[i, j, k+1]) if k + 1 < M else 0

                t = int(C1[i, j, k])
                t = sub_bytes_u32_numba(t, inv_sbox_layers[k])
                t ^= int(ks2[i, j, k])
                s = rotr32_numba(t, 9)

                orig = (s
                        - rotl32_numba(dn, 5)
                        - rotl32_numba(rt, 13)
                        - rotl32_numba(nxt, 17)
                        - int(mask2[i, j, k])) & MOD_MASK

                C1[i, j, k] = np.uint32(orig)
    return C1

@njit(cache=True)
def u32_undo_pass1_backward_numba(C1, mask1, ks1, inv_sbox_layers, depth_mask):
    H, W, M = C1.shape
    U = np.empty_like(C1)
    for i in range(H - 1, -1, -1):
        for j in range(W - 1, -1, -1):
            for k in range(M - 1, -1, -1):
                if depth_mask[k] == 0:
                    U[i, j, k] = C1[i, j, k]
                    continue
                up = int(C1[i-1, j, k]) if i > 0 else 0
                left = int(C1[i, j-1, k]) if j > 0 else 0
                prev = int(C1[i, j, k-1]) if k > 0 else 0

                t = int(C1[i, j, k])
                t = sub_bytes_u32_numba(t, inv_sbox_layers[k])
                t ^= int(ks1[i, j, k])
                s = rotr32_numba(t, 3)

                u = (s
                     - rotl32_numba(up, 7)
                     - rotl32_numba(left, 11)
                     - rotl32_numba(prev, 19)
                     - int(mask1[i, j, k])) & MOD_MASK

                U[i, j, k] = np.uint32(u)
    return U

def make_depth_mask(M: int, enc_ratio: float) -> np.ndarray:
    enc_ratio = float(enc_ratio)
    enc_ratio = max(0.0, min(1.0, enc_ratio))
    k = int(round(enc_ratio * M))
    k = max(0, min(M, k))
    m = np.zeros(M, dtype=np.uint8)
    if k > 0:
        m[:k] = 1
    return m

def u32_diffuse_forward(U: np.ndarray, key: str, no_sbox: bool, static_sbox: bool, enc_ratio: float, use_numba=True):
    H, W, M = U.shape
    depth_mask = make_depth_mask(M, enc_ratio)

    if no_sbox:
        sbox1_layers, _ = build_identity_sbox_layers(M)
        sbox2_layers, _ = build_identity_sbox_layers(M)
    elif static_sbox:
        sbox1_layers, _ = build_static_sbox_layers(key, M, "sbox_pass1")
        sbox2_layers, _ = build_static_sbox_layers(key, M, "sbox_pass2")
    else:
        sbox1_layers, _ = build_sbox_layers_from_chaos(key, M, "sbox_pass1")
        sbox2_layers, _ = build_sbox_layers_from_chaos(key, M, "sbox_pass2")

    mask1 = u32_keystream(key + "|mask1|", U.shape)
    ks1   = u32_keystream(key + "|ks1|",   U.shape)
    mask2 = u32_keystream(key + "|mask2|", U.shape)
    ks2   = u32_keystream(key + "|ks2|",   U.shape)

    if use_numba and NUMBA_AVAILABLE:
        C = u32_pass1_forward_numba(U, mask1, ks1, sbox1_layers, depth_mask)
        C = u32_pass2_backward_numba(C, mask2, ks2, sbox2_layers, depth_mask)
        return C.astype(np.uint32)

    C = U.copy().astype(np.uint32)

    # pass1 forward
    for i in range(H):
        for j in range(W):
            for k in range(M):
                if depth_mask[k] == 0:
                    C[i, j, k] = U[i, j, k]
                    continue
                up   = int(C[i-1, j, k]) if i > 0 else 0
                left = int(C[i, j-1, k]) if j > 0 else 0
                prev = int(C[i, j, k-1]) if k > 0 else 0
                s = (int(U[i, j, k])
                     + rotl32_py(up, 7)
                     + rotl32_py(left, 11)
                     + rotl32_py(prev, 19)
                     + int(mask1[i, j, k])) & MOD_MASK
                t = rotl32_py(s, 3) ^ int(ks1[i, j, k])
                t = sub_bytes_u32_py(t, sbox1_layers[k])
                C[i, j, k] = np.uint32(t)

    # pass2 backward
    for i in range(H-1, -1, -1):
        for j in range(W-1, -1, -1):
            for k in range(M-1, -1, -1):
                if depth_mask[k] == 0:
                    continue
                dn  = int(C[i+1, j, k]) if i+1 < H else 0
                rt  = int(C[i, j+1, k]) if j+1 < W else 0
                nxt = int(C[i, j, k+1]) if k+1 < M else 0
                s = (int(C[i, j, k])
                     + rotl32_py(dn, 5)
                     + rotl32_py(rt, 13)
                     + rotl32_py(nxt, 17)
                     + int(mask2[i, j, k])) & MOD_MASK
                t = rotl32_py(s, 9) ^ int(ks2[i, j, k])
                t = sub_bytes_u32_py(t, sbox2_layers[k])
                C[i, j, k] = np.uint32(t)

    return C

def u32_diffuse_inverse(Cu32: np.ndarray, key: str, no_sbox: bool, static_sbox: bool, enc_ratio: float, use_numba=True):
    H, W, M = Cu32.shape
    depth_mask = make_depth_mask(M, enc_ratio)

    if no_sbox:
        _, inv1_layers = build_identity_sbox_layers(M)
        _, inv2_layers = build_identity_sbox_layers(M)
    elif static_sbox:
        _, inv1_layers = build_static_sbox_layers(key, M, "sbox_pass1")
        _, inv2_layers = build_static_sbox_layers(key, M, "sbox_pass2")
    else:
        _, inv1_layers = build_sbox_layers_from_chaos(key, M, "sbox_pass1")
        _, inv2_layers = build_sbox_layers_from_chaos(key, M, "sbox_pass2")

    mask1 = u32_keystream(key + "|mask1|", Cu32.shape)
    ks1   = u32_keystream(key + "|ks1|",   Cu32.shape)
    mask2 = u32_keystream(key + "|mask2|", Cu32.shape)
    ks2   = u32_keystream(key + "|ks2|",   Cu32.shape)

    if use_numba and NUMBA_AVAILABLE:
        C1 = u32_undo_pass2_forward_numba(Cu32, mask2, ks2, inv2_layers, depth_mask)
        U  = u32_undo_pass1_backward_numba(C1,  mask1, ks1, inv1_layers, depth_mask)
        return U.astype(np.uint32)

    C1 = Cu32.copy().astype(np.uint32)

    # undo pass2 forward scan
    for i in range(H):
        for j in range(W):
            for k in range(M):
                if depth_mask[k] == 0:
                    continue
                dn  = int(C1[i+1, j, k]) if i+1 < H else 0
                rt  = int(C1[i, j+1, k]) if j+1 < W else 0
                nxt = int(C1[i, j, k+1]) if k+1 < M else 0

                t = int(C1[i, j, k])
                t = sub_bytes_u32_py(t, inv2_layers[k])
                t ^= int(ks2[i, j, k])
                s = rotr32_py(t, 9)

                orig = (s
                        - rotl32_py(dn, 5)
                        - rotl32_py(rt, 13)
                        - rotl32_py(nxt, 17)
                        - int(mask2[i, j, k])) & MOD_MASK
                C1[i, j, k] = np.uint32(orig)

    # undo pass1 backward scan
    U = np.empty_like(C1, dtype=np.uint32)
    for i in range(H-1, -1, -1):
        for j in range(W-1, -1, -1):
            for k in range(M-1, -1, -1):
                if depth_mask[k] == 0:
                    U[i, j, k] = C1[i, j, k]
                    continue
                up   = int(C1[i-1, j, k]) if i > 0 else 0
                left = int(C1[i, j-1, k]) if j > 0 else 0
                prev = int(C1[i, j, k-1]) if k > 0 else 0

                t = int(C1[i, j, k])
                t = sub_bytes_u32_py(t, inv1_layers[k])
                t ^= int(ks1[i, j, k])
                s = rotr32_py(t, 3)

                u = (s
                     - rotl32_py(up, 7)
                     - rotl32_py(left, 11)
                     - rotl32_py(prev, 19)
                     - int(mask1[i, j, k])) & MOD_MASK
                U[i, j, k] = np.uint32(u)

    return U


# ============================================================
# Cipher images (invertible RGBA + non-invertible uint8 view)
# ============================================================

def cipher_uint8_view_from_Cu32(Cu32: np.ndarray, out_H: int, out_W: int):
    x = Cu32.astype(np.uint32)
    u8 = ((x >> 24) ^ (x >> 16) ^ (x >> 8) ^ x).astype(np.uint8)
    flat = u8.flatten()
    if flat.size == out_H * out_W:
        return flat.reshape((out_H, out_W))
    gh, gw, m = u8.shape
    return u8.reshape((gh, gw * m))

def save_cipher_rgba(Cu32: np.ndarray, out_path: str):
    H, W, M = Cu32.shape
    words = Cu32.reshape(-1).astype(np.uint32)
    b0 = ((words >> 24) & 0xFF).astype(np.uint8)
    b1 = ((words >> 16) & 0xFF).astype(np.uint8)
    b2 = ((words >> 8) & 0xFF).astype(np.uint8)
    b3 = (words & 0xFF).astype(np.uint8)
    rgba = np.stack([b0, b1, b2, b3], axis=1)
    img_rgba = rgba.reshape(H, W * M, 4)
    # Pillow 13 deprecates mode arg; let it infer from shape/dtype
    Image.fromarray(img_rgba).save(out_path)

def load_cipher_rgba_to_Cu32(path: str, H_: int, W_: int, M_: int) -> np.ndarray:
    im = Image.open(path).convert("RGBA")
    arr = np.array(im, dtype=np.uint8)
    if arr.shape[0] != H_ or arr.shape[1] != W_ * M_ or arr.shape[2] != 4:
        raise ValueError(f"cipher_rgba shape mismatch: got {arr.shape}, expect ({H_},{W_*M_},4)")
    flat = arr.reshape(-1, 4).astype(np.uint32)
    words = (flat[:,0] << 24) | (flat[:,1] << 16) | (flat[:,2] << 8) | flat[:,3]
    return words.reshape(H_, W_, M_).astype(np.uint32)


# ============================================================
# FISTA solver
# ============================================================

def power_iteration_L(Phi: np.ndarray, iters: int = 10):
    n = Phi.shape[1]
    v = np.random.default_rng(0).standard_normal(n)
    v /= (np.linalg.norm(v) + 1e-12)
    for _ in range(iters):
        v = Phi.T @ (Phi @ v)
        v /= (np.linalg.norm(v) + 1e-12)
    w = Phi @ v
    return float(np.dot(w, w))

def soft_threshold(x: np.ndarray, lam: float):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)

def fista_l1(Phi: np.ndarray, y: np.ndarray, lam: float, max_iter: int = 120, tol: float = 1e-5):
    n = Phi.shape[1]
    x = np.zeros(n, dtype=np.float64)
    z = x.copy()
    t = 1.0
    L = power_iteration_L(Phi, iters=10) + 1e-12
    invL = 1.0 / L

    prev = 1e30
    for _ in range(max_iter):
        grad = Phi.T @ (Phi @ z - y)
        x_new = soft_threshold(z - invL * grad, lam * invL)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        t = t_new
        x = x_new

        obj = 0.5 * np.linalg.norm(Phi @ x - y) ** 2 + lam * np.sum(np.abs(x))
        if abs(prev - obj) / (abs(prev) + 1e-12) < tol:
            break
        prev = obj

    return x


# ============================================================
# Crypto state + core
# ============================================================

@dataclass
class CryptoState:
    H: int
    W: int
    B: int
    m_meas: int
    grid_h: int
    grid_w: int
    y_min: float
    y_max: float
    diff_abc: tuple
    mask_start: int
    chaos_need: int
    static_phi: bool

    def to_npz(self, path: str):
        np.savez(
            path,
            H=self.H, W=self.W, B=self.B,
            m_meas=self.m_meas,
            grid_h=self.grid_h, grid_w=self.grid_w,
            y_min=self.y_min, y_max=self.y_max,
            a=self.diff_abc[0], b=self.diff_abc[1], c=self.diff_abc[2],
            mask_start=self.mask_start,
            chaos_need=self.chaos_need,
            static_phi=int(self.static_phi),
        )

    @staticmethod
    def from_npz(path: str):
        d = np.load(path, allow_pickle=False)
        return CryptoState(
            H=int(d["H"]), W=int(d["W"]), B=int(d["B"]),
            m_meas=int(d["m_meas"]),
            grid_h=int(d["grid_h"]), grid_w=int(d["grid_w"]),
            y_min=float(d["y_min"]), y_max=float(d["y_max"]),
            diff_abc=(float(d["a"]), float(d["b"]), float(d["c"])),
            mask_start=int(d["mask_start"]),
            chaos_need=int(d["chaos_need"]),
            static_phi=bool(int(d["static_phi"])),
        )

class CSChaosCrypto:
    def __init__(self, key: str, B=8, m_meas=64, static_phi=False, use_numba=True):
        self.key = key
        self.B = int(B)
        self.n = 64
        self.m_meas = int(m_meas)
        self.static_phi = bool(static_phi)
        self.use_numba = bool(use_numba)

        self._Phi_static_cache = None
        if self.static_phi:
            self._Phi_static_cache = phi_static(self.key, self.m_meas, self.n)

    def _get_phi(self, zs, bi):
        if self.static_phi:
            return self._Phi_static_cache
        return phi_from_key_z(self.key, zs[bi], block_id=bi, m_meas=self.m_meas, n=self.n)

    def encrypt_prequant_D(self, img_u8: np.ndarray, float_diffusion=True):
        Pn = img_u8.astype(np.float64) / 255.0
        H, W = Pn.shape
        blocks, (gh, gw) = blockify(Pn, self.B)
        N = blocks.shape[0]

        chaos_need = max(N + 4096, gh * gw * self.m_meas + 4096)
        _, _, zs = spcmm_generate_xyz(self.key, chaos_need)

        Omegas = np.zeros((N, self.n), dtype=np.float64)
        for i in range(N):
            C = dct2(blocks[i])
            Omegas[i] = C.flatten()

        Y_meas = np.zeros((N, self.m_meas), dtype=np.float64)
        for bi in range(N):
            Phi = self._get_phi(zs, bi)
            Y_meas[bi] = Phi @ Omegas[bi]

        Y = Y_meas.reshape((gh, gw, self.m_meas))
        y_min = float(Y.min())
        y_max = float(Y.max())
        denom = (y_max - y_min) if (y_max - y_min) > 1e-12 else 1.0
        Yn = frac_arr((Y - y_min) / denom)

        mask_start = 1024
        flat_mask = zs[mask_start:mask_start + gh * gw * self.m_meas]
        mask = frac_arr(flat_mask.reshape((gh, gw, self.m_meas)))

        a, b, c = (0.21, 0.33, 0.27)
        if float_diffusion:
            D = diffusion3d_forward(Yn, mask, a=a, b=b, c=c, use_numba=self.use_numba)
        else:
            D = Yn.copy()

        state = CryptoState(
            H=H, W=W, B=self.B,
            m_meas=self.m_meas,
            grid_h=gh, grid_w=gw,
            y_min=y_min, y_max=y_max,
            diff_abc=(a, b, c),
            mask_start=mask_start,
            chaos_need=chaos_need,
            static_phi=self.static_phi,
        )
        return D.astype(np.float64), state

    def decrypt_from_prequant_D(self, D: np.ndarray, state: CryptoState,
                               fista_lam: float, fista_iter: int,
                               float_diffusion=True):
        gh, gw = state.grid_h, state.grid_w
        assert D.shape == (gh, gw, state.m_meas)

        D = frac_arr(D.astype(np.float64))
        _, _, zs = spcmm_generate_xyz(self.key, state.chaos_need)

        flat_mask = zs[state.mask_start:state.mask_start + gh * gw * state.m_meas]
        mask = frac_arr(flat_mask.reshape((gh, gw, state.m_meas)))

        a, b, c = state.diff_abc
        if float_diffusion:
            Yn = diffusion3d_inverse(D, mask, a=a, b=b, c=c, use_numba=self.use_numba)
        else:
            Yn = D.copy()

        denom = (state.y_max - state.y_min) if (state.y_max - state.y_min) > 1e-12 else 1.0
        Y = Yn * denom + state.y_min

        Y_meas = Y.reshape((-1, state.m_meas))
        N = Y_meas.shape[0]
        Omegas_hat = np.zeros((N, self.n), dtype=np.float64)

        # phi cache for static case
        Phi_static_cache = None
        if state.static_phi:
            Phi_static_cache = phi_static(self.key, state.m_meas, self.n)

        for bi in range(N):
            if state.static_phi:
                Phi = Phi_static_cache
            else:
                Phi = phi_from_key_z(self.key, zs[bi], block_id=bi, m_meas=state.m_meas, n=self.n)
            y = Y_meas[bi]

            if state.m_meas == self.n:
                Omegas_hat[bi] = Phi.T @ y
            else:
                Omegas_hat[bi] = fista_l1(Phi, y, lam=fista_lam, max_iter=fista_iter, tol=1e-5)

        blocks_rec = []
        for i in range(N):
            C_rec = Omegas_hat[i].reshape((self.B, self.B))
            blk = idct2(C_rec)
            blocks_rec.append(blk)
        blocks_rec = np.array(blocks_rec, dtype=np.float64)

        Pn_hat = unblockify(blocks_rec, (gh, gw), B=self.B)
        Pn_hat = np.clip(Pn_hat, 0, 1)
        return (Pn_hat * 255.0 + 0.5).astype(np.uint8)


# ============================================================
# Full measurement module
# ============================================================

def mae_u8(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.int32) - b.astype(np.int32))))

def mse_u8(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(np.float64)
    bb = b.astype(np.float64)
    return float(np.mean((aa - bb) ** 2))

def psnr_u8(a: np.ndarray, b: np.ndarray) -> float:
    m = mse_u8(a, b)
    if m < 1e-12:
        return 99.0
    return float(10.0 * np.log10((255.0 ** 2) / m))

def ssim_u8(a: np.ndarray, b: np.ndarray) -> float:
    if ssim_skimage is not None:
        return float(ssim_skimage(a, b, data_range=255))
    a = a.astype(np.float64); b = b.astype(np.float64)
    mu_x = a.mean(); mu_y = b.mean()
    var_x = a.var(); var_y = b.var()
    cov_xy = ((a - mu_x) * (b - mu_y)).mean()
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    return float(((2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)) /
                 ((mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2)))

def entropy_u8(img: np.ndarray) -> float:
    hist = np.bincount(img.flatten(), minlength=256).astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def npcr_u8(c1: np.ndarray, c2: np.ndarray) -> float:
    return float(np.mean(c1 != c2) * 100.0)

def uaci_u8(c1: np.ndarray, c2: np.ndarray) -> float:
    return float(np.mean(np.abs(c1.astype(np.int32) - c2.astype(np.int32))) / 255.0 * 100.0)

def corr_adjacent(img: np.ndarray, mode: str = "h", samples: int = 200000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    H, W = img.shape
    if W < 2 or H < 2:
        return 0.0
    samples = int(min(samples, H * W))
    if mode == "h":
        xs = rng.integers(0, H, size=samples); ys = rng.integers(0, W - 1, size=samples)
        p1 = img[xs, ys].astype(np.float64); p2 = img[xs, ys + 1].astype(np.float64)
    elif mode == "v":
        xs = rng.integers(0, H - 1, size=samples); ys = rng.integers(0, W, size=samples)
        p1 = img[xs, ys].astype(np.float64); p2 = img[xs + 1, ys].astype(np.float64)
    elif mode == "d":
        xs = rng.integers(0, H - 1, size=samples); ys = rng.integers(0, W - 1, size=samples)
        p1 = img[xs, ys].astype(np.float64); p2 = img[xs + 1, ys + 1].astype(np.float64)
    else:
        raise ValueError("mode must be 'h','v','d'")
    p1m = p1 - p1.mean(); p2m = p2 - p2.mean()
    denom = (np.sqrt(np.mean(p1m**2)) * np.sqrt(np.mean(p2m**2)) + 1e-12)
    return float(np.mean(p1m * p2m) / denom)

def save_histogram(img_u8: np.ndarray, out_path: str, title: str):
    hist = np.bincount(img_u8.flatten(), minlength=256)
    x = np.arange(256)
    plt.figure()
    plt.plot(x, hist)
    plt.title(title)
    plt.xlabel("Gray level")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_corr_scatter(img_u8: np.ndarray, out_path: str, mode: str, samples: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    H, W = img_u8.shape
    if W < 2 or H < 2:
        return
    samples = int(min(samples, H * W))
    if mode == "h":
        xs = rng.integers(0, H, size=samples); ys = rng.integers(0, W - 1, size=samples)
        p1 = img_u8[xs, ys]; p2 = img_u8[xs, ys + 1]
        title = "Adjacent Correlation Scatter (Horizontal)"
    elif mode == "v":
        xs = rng.integers(0, H - 1, size=samples); ys = rng.integers(0, W, size=samples)
        p1 = img_u8[xs, ys]; p2 = img_u8[xs + 1, ys]
        title = "Adjacent Correlation Scatter (Vertical)"
    elif mode == "d":
        xs = rng.integers(0, H - 1, size=samples); ys = rng.integers(0, W - 1, size=samples)
        p1 = img_u8[xs, ys]; p2 = img_u8[xs + 1, ys + 1]
        title = "Adjacent Correlation Scatter (Diagonal)"
    else:
        raise ValueError("mode must be 'h','v','d'")
    plt.figure()
    plt.scatter(p1, p2, s=2, alpha=0.35)
    plt.title(title)
    plt.xlabel("Pixel value")
    plt.ylabel("Neighbor value")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ============================================================
# Ablation config
# ============================================================

def parse_ablate(mode: str):
    """
    Returns:
      float_diffusion (bool),
      u32_arx (bool),
      no_sbox (bool),
      static_sbox (bool)
    """
    mode = (mode or "full").strip()
    # defaults (full)
    float_diffusion = True
    u32_arx = True
    no_sbox = False
    static_sbox = False

    if mode == "full":
        pass
    elif mode == "baseline_cs":
        float_diffusion = False
        u32_arx = False
        no_sbox = True
        static_sbox = False
    elif mode == "no_float_diff":
        float_diffusion = False
    elif mode == "no_u32_arx":
        u32_arx = False
        no_sbox = True
        static_sbox = False
    elif mode == "no_sbox":
        no_sbox = True
        static_sbox = False
    elif mode == "static_sbox":
        no_sbox = False
        static_sbox = True
    else:
        # unknown -> treat as full
        pass

    return float_diffusion, u32_arx, no_sbox, static_sbox


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=str, default="1.png")
    ap.add_argument("--out", type=str, default="outputs_fista_full")
    ap.add_argument("--key", type=str, default="my-secret-key-2026")
    ap.add_argument("--wrong_key", type=str, default="my-secret-key-2026_wrong")
    ap.add_argument("--cr", type=float, default=0.5, help="compression rate m/64")
    ap.add_argument("--lam", type=float, default=0.01, help="FISTA lambda (L1 weight)")
    ap.add_argument("--iter", type=int, default=150, help="FISTA max iterations per block")

    # ablations + toggles (match your current prints)
    ap.add_argument("--ablate", type=str, default="full",
                    help="baseline_cs|no_float_diff|no_u32_arx|no_sbox|static_sbox|full")
    ap.add_argument("--static_phi", type=int, default=0, help="1=use same Phi for all blocks")
    ap.add_argument("--enc_ratio", type=float, default=1.0, help="encrypt only first ratio of depth channels (0..1)")

    # perf
    ap.add_argument("--no_numba", type=int, default=0, help="1=disable numba even if available")

    # NEW: decrypt-only / external cipher
    ap.add_argument("--cipher_in", type=str, default="", help="external cipher_rgba.png for decrypt_only")
    ap.add_argument("--state_in", type=str, default="", help="state.npz path for decrypt_only")
    ap.add_argument("--decrypt_only", type=int, default=0)
    ap.add_argument("--encrypt_only", type=int, default=0)

    args = ap.parse_args()

    # core params
    B = 8
    n = 64
    cr = float(args.cr)
    cr = max(1.0 / n, min(1.0, cr))
    m_meas = int(np.clip(int(round(cr * n)), 1, n))

    ensure_dir(args.out)

    # numba switch
    use_numba = (args.no_numba == 0)

    # ablation config
    float_diffusion, u32_arx, no_sbox, static_sbox = parse_ablate(args.ablate)
    static_phi_flag = bool(args.static_phi)

    KEY = args.key
    WRONG_KEY = args.wrong_key
    KEY_FLIP = flip_one_bit_in_key(KEY)

    # ---- Load image (needed even for decrypt_only to compute metrics and for baseline refs)
    img = Image.open(args.img).convert("L")
    img = np.array(img, dtype=np.uint8)

    H, W = img.shape
    H2, W2 = (H // B) * B, (W // B) * B
    img = img[:H2, :W2]
    H, W = img.shape

    # ---- Print config (match your style)
    print("=== Config ===")
    print(f"image: {args.img}  size: {H}x{W}")
    print(f"block: {B}x{B}  n=64")
    print(f"compression rate cr=m/n: {cr:.4f}  -> m_meas={m_meas}")
    print(f"solver: fista   lam: {args.lam}   iter: {args.iter}")
    print(f"Numba enabled: {bool(NUMBA_AVAILABLE and use_numba)}")
    print(f"ablate mode: {args.ablate}")
    print(f"static_phi: {static_phi_flag}")
    print(f"float_diffusion: {float_diffusion}")
    print(f"u32_arx: {u32_arx}  no_sbox={no_sbox}  enc_ratio={args.enc_ratio}")
    print("")

    crypto = CSChaosCrypto(KEY, B=B, m_meas=m_meas, static_phi=static_phi_flag, use_numba=use_numba)

    # ============================================================
    # decrypt-only path
    # ============================================================
    if args.decrypt_only:
        cipher_rgba_path = args.cipher_in.strip()
        if not cipher_rgba_path:
            raise ValueError("--decrypt_only requires --cipher_in path/to/cipher_rgba.png")

        # resolve state.npz
        state_path = args.state_in.strip()
        if not state_path:
            cand1 = os.path.join(os.path.dirname(cipher_rgba_path), "state.npz")
            cand2 = os.path.join(args.out, "state.npz")
            state_path = cand1 if os.path.exists(cand1) else cand2
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"state.npz not found. Provide --state_in explicitly. Tried: {state_path}")

        state = CryptoState.from_npz(state_path)
        gh, gw, mm = state.grid_h, state.grid_w, state.m_meas

        # decrypt from attacked cipher
        Cu32_from_img = load_cipher_rgba_to_Cu32(cipher_rgba_path, gh, gw, mm)

        # if u32_arx disabled in this mode, treat cipher as raw U (still stored in RGBA)
        t2 = time.perf_counter()
        if u32_arx:
            U_test = u32_diffuse_inverse(
                Cu32_from_img, KEY,
                no_sbox=no_sbox, static_sbox=static_sbox,
                enc_ratio=args.enc_ratio,
                use_numba=use_numba
            )
        else:
            U_test = Cu32_from_img.astype(np.uint32)

        D_back = float01_from_u32(U_test)

        rec = crypto.decrypt_from_prequant_D(
            D_back, state,
            fista_lam=args.lam, fista_iter=args.iter,
            float_diffusion=float_diffusion
        )
        t3 = time.perf_counter()
        dec_time = t3 - t2

        # metrics
        MAE = mae_u8(rec, img)
        PSNR = psnr_u8(rec, img)
        SSIM = ssim_u8(rec, img)

        ensure_dir(args.out)
        Image.fromarray(img).save(os.path.join(args.out, "plain.png"))
        Image.fromarray(rec).save(os.path.join(args.out, "decrypted.png"))

        # report
        lines = []
        lines.append("=== Decrypt ONLY ===")
        lines.append(f"cipher_in: {cipher_rgba_path}")
        lines.append(f"state_in: {state_path}")
        lines.append(f"Decrypt time: {dec_time:.6f}s")
        lines.append("")
        lines.append("=== Reconstruction quality (plain vs decrypted) ===")
        lines.append(f"MAE:  {MAE}")
        lines.append(f"PSNR: {PSNR}")
        lines.append(f"SSIM: {SSIM}")
        report = "\n".join(lines)
        print(report)
        with open(os.path.join(args.out, "report.txt"), "w", encoding="utf-8") as f:
            f.write(report)
        return

    # ============================================================
    # Normal encrypt (and optionally decrypt/eval)
    # ============================================================
    t0 = time.perf_counter()

    D, state = crypto.encrypt_prequant_D(img, float_diffusion=float_diffusion)

    # save state so robustness experiments can decrypt external ciphers
    state_path = os.path.join(args.out, "state.npz")
    state.to_npz(state_path)

    U = u32_from_float01(D)

    if u32_arx:
        Cu32 = u32_diffuse_forward(
            U, KEY,
            no_sbox=no_sbox, static_sbox=static_sbox,
            enc_ratio=args.enc_ratio,
            use_numba=use_numba
        )
    else:
        Cu32 = U.copy().astype(np.uint32)

    t1 = time.perf_counter()
    enc_time = t1 - t0

    plain_bytes = img.size
    enc_thr = (plain_bytes / (1024 * 1024)) / max(enc_time, 1e-12)

    cipher_rgba_path = os.path.join(args.out, "cipher_rgba.png")
    save_cipher_rgba(Cu32, cipher_rgba_path)

    cipher_u8 = cipher_uint8_view_from_Cu32(Cu32, H, W)
    cipher_u8_path = os.path.join(args.out, "cipher_uint8.png")
    Image.fromarray(cipher_u8).save(cipher_u8_path)

    print("=== Encrypt ===")
    print(f"D shape: {D.shape} float64 range: {float(D.min())} {float(D.max())}")
    print(f"Cu32 shape: {Cu32.shape} uint32 (REAL ciphertext container)")
    print(f"Encrypt time: {enc_time:.6f}s  throughput: {enc_thr:.3f} MiB/s")
    print(f"cipher_rgba.png saved: {cipher_rgba_path} (INVERTIBLE FINAL CIPHER IMAGE)\n")

    if args.encrypt_only:
        # minimal report for encrypt-only
        with open(os.path.join(args.out, "report.txt"), "w", encoding="utf-8") as f:
            f.write(f"Encrypt only\ncipher_rgba: {cipher_rgba_path}\nstate: {state_path}\n")
        return

    # ============================================================
    # Decrypt from cipher_rgba.png (correct key)
    # ============================================================
    gh, gw, mm = state.grid_h, state.grid_w, state.m_meas
    Cu32_from_img = load_cipher_rgba_to_Cu32(cipher_rgba_path, gh, gw, mm)
    cu32_img_mismatch = int(np.count_nonzero(Cu32_from_img != Cu32))

    t2 = time.perf_counter()
    if u32_arx:
        U_test = u32_diffuse_inverse(
            Cu32_from_img, KEY,
            no_sbox=no_sbox, static_sbox=static_sbox,
            enc_ratio=args.enc_ratio,
            use_numba=use_numba
        )
    else:
        U_test = Cu32_from_img.astype(np.uint32)

    D_back = float01_from_u32(U_test)

    rec = crypto.decrypt_from_prequant_D(
        D_back, state,
        fista_lam=args.lam, fista_iter=args.iter,
        float_diffusion=float_diffusion
    )
    t3 = time.perf_counter()

    dec_time = t3 - t2
    dec_thr = (plain_bytes / (1024 * 1024)) / max(dec_time, 1e-12)
    mismatch = int(np.count_nonzero(U_test != U))

    print("=== Decrypt FROM cipher_rgba.png (correct key) ===")
    print(f"Cu32(image) mismatch count: {cu32_img_mismatch} (should be 0)")
    print(f"u32 invertibility mismatch count: {mismatch} (should be 0)")
    print(f"Decrypt time: {dec_time:.6f}s  throughput: {dec_thr:.3f} MiB/s\n")

    # Save base outputs
    Image.fromarray(img).save(os.path.join(args.out, "plain.png"))
    Image.fromarray(rec).save(os.path.join(args.out, "decrypted.png"))
    diff = np.abs(rec.astype(np.int32) - img.astype(np.int32))
    diff_vis = np.clip(diff, 0, 255).astype(np.uint8)
    Image.fromarray(diff_vis).save(os.path.join(args.out, "abs_diff.png"))

    np.save(os.path.join(args.out, "cipher_u32.npy"), Cu32)
    np.save(os.path.join(args.out, "cipher_D.npy"), D)

    # Reconstruction quality
    MAE = mae_u8(rec, img)
    PSNR = psnr_u8(rec, img)
    SSIM = ssim_u8(rec, img)

    print("=== Reconstruction quality (plain vs decrypted) ===")
    print(f"MAE:  {MAE}")
    print(f"PSNR: {PSNR}")
    print(f"SSIM: {SSIM}\n")

    # Cipher stats (cipher_uint8 view)
    ENT = entropy_u8(cipher_u8)
    CH = corr_adjacent(cipher_u8, "h", seed=1)
    CV = corr_adjacent(cipher_u8, "v", seed=2)
    CD = corr_adjacent(cipher_u8, "d", seed=3)

    print("=== Cipher statistics ON cipher_uint8.png (view derived from Cu32) ===")
    print(f"cipher_uint8 shape: {cipher_u8.shape}")
    print(f"Entropy: {ENT:.6f} bits")
    print(f"AdjCorr-H: {CH:.6f}")
    print(f"AdjCorr-V: {CV:.6f}")
    print(f"AdjCorr-D: {CD:.6f}\n")

    # plots
    save_histogram(img, os.path.join(args.out, "hist_plain.png"), "Histogram (Plain)")
    save_histogram(cipher_u8, os.path.join(args.out, "hist_cipher.png"), "Histogram (Cipher uint8 view)")
    save_histogram(rec, os.path.join(args.out, "hist_decrypted.png"), "Histogram (Decrypted)")

    save_corr_scatter(cipher_u8, os.path.join(args.out, "corr_cipher_h.png"), "h", seed=10)
    save_corr_scatter(cipher_u8, os.path.join(args.out, "corr_cipher_v.png"), "v", seed=11)
    save_corr_scatter(cipher_u8, os.path.join(args.out, "corr_cipher_d.png"), "d", seed=12)

    # Differential test (flip 1 plaintext pixel) ON cipher_uint8 view
    img2 = img.copy()
    img2[0, 0] = np.uint8((int(img2[0, 0]) + 1) % 256)
    D2, _ = crypto.encrypt_prequant_D(img2, float_diffusion=float_diffusion)
    U2 = u32_from_float01(D2)
    if u32_arx:
        Cu32_2 = u32_diffuse_forward(U2, KEY, no_sbox=no_sbox, static_sbox=static_sbox, enc_ratio=args.enc_ratio, use_numba=use_numba)
    else:
        Cu32_2 = U2.copy().astype(np.uint32)
    cipher_u8_2 = cipher_uint8_view_from_Cu32(Cu32_2, H, W)

    NPCR = npcr_u8(cipher_u8, cipher_u8_2)
    UACI = uaci_u8(cipher_u8, cipher_u8_2)
    cipher_diff = np.abs(cipher_u8.astype(np.int16) - cipher_u8_2.astype(np.int16))
    Image.fromarray(np.clip(cipher_diff, 0, 255).astype(np.uint8)).save(os.path.join(args.out, "cipher_diff_abs.png"))

    print("=== Differential test (flip 1 plaintext pixel) ON cipher_uint8 view ===")
    print(f"NPCR: {NPCR:.6f}%")
    print(f"UACI: {UACI:.6f}%\n")

    # Key sensitivity test (flip 1 key bit)
    if u32_arx:
        Cu32_k = u32_diffuse_forward(U, KEY_FLIP, no_sbox=no_sbox, static_sbox=static_sbox, enc_ratio=args.enc_ratio, use_numba=use_numba)
    else:
        Cu32_k = U.copy().astype(np.uint32)
    cipher_u8_k = cipher_uint8_view_from_Cu32(Cu32_k, H, W)
    NPCR_key = npcr_u8(cipher_u8, cipher_u8_k)
    UACI_key = uaci_u8(cipher_u8, cipher_u8_k)
    Image.fromarray(cipher_u8_k).save(os.path.join(args.out, "cipher_uint8_keyflip.png"))

    print("=== Key sensitivity test (flip 1 key bit) ON cipher_uint8 view ===")
    print(f"Key original: {KEY}")
    print(f"Key flipped:  {KEY_FLIP}")
    print(f"NPCR_key: {NPCR_key:.6f}%")
    print(f"UACI_key: {UACI_key:.6f}%\n")

    # Wrong-key decryption (from cipher_rgba.png)
    if u32_arx:
        U_wrong = u32_diffuse_inverse(Cu32_from_img, WRONG_KEY, no_sbox=no_sbox, static_sbox=static_sbox, enc_ratio=args.enc_ratio, use_numba=use_numba)
    else:
        U_wrong = Cu32_from_img.astype(np.uint32)
    D_wrong = float01_from_u32(U_wrong)
    crypto_wrong = CSChaosCrypto(WRONG_KEY, B=B, m_meas=m_meas, static_phi=static_phi_flag, use_numba=use_numba)
    rec_wrong = crypto_wrong.decrypt_from_prequant_D(D_wrong, state, fista_lam=args.lam, fista_iter=args.iter, float_diffusion=float_diffusion)
    Image.fromarray(rec_wrong).save(os.path.join(args.out, "decrypted_wrong_key.png"))

    MAE_wrong = mae_u8(rec_wrong, img)
    PSNR_wrong = psnr_u8(rec_wrong, img)
    SSIM_wrong = ssim_u8(rec_wrong, img)

    print("=== Wrong-key decryption quality (from cipher_rgba.png) ===")
    print(f"Wrong key: {WRONG_KEY}")
    print(f"MAE_wrong:  {MAE_wrong}")
    print(f"PSNR_wrong: {PSNR_wrong}")
    print(f"SSIM_wrong: {SSIM_wrong}\n")

    # Report
    lines = []
    lines.append("=== Config ===")
    lines.append(f"image: {args.img}  size: {H}x{W}")
    lines.append(f"block: {B}x{B}  n=64")
    lines.append(f"compression rate cr=m/n: {cr:.4f}  -> m_meas={m_meas}")
    lines.append(f"solver: fista   lam: {args.lam}   iter: {args.iter}")
    lines.append(f"Numba enabled: {bool(NUMBA_AVAILABLE and use_numba)}")
    lines.append(f"ablate mode: {args.ablate}")
    lines.append(f"static_phi: {static_phi_flag}")
    lines.append(f"float_diffusion: {float_diffusion}")
    lines.append(f"u32_arx: {u32_arx}  no_sbox={no_sbox}  enc_ratio={args.enc_ratio}")
    lines.append(f"state saved: {state_path}\n")

    lines.append("=== Encrypt ===")
    lines.append(f"D shape: {D.shape} float64 range: {float(D.min())} {float(D.max())}")
    lines.append(f"Cu32 shape: {Cu32.shape} uint32 (REAL ciphertext container)")
    lines.append(f"Encrypt time: {enc_time:.6f}s  throughput: {enc_thr:.3f} MiB/s")
    lines.append(f"cipher_rgba.png saved: {cipher_rgba_path} (INVERTIBLE FINAL CIPHER IMAGE)\n")

    lines.append("=== Decrypt FROM cipher_rgba.png (correct key) ===")
    lines.append(f"Cu32(image) mismatch count: {cu32_img_mismatch} (should be 0)")
    lines.append(f"u32 invertibility mismatch count: {mismatch} (should be 0)")
    lines.append(f"Decrypt time: {dec_time:.6f}s  throughput: {dec_thr:.3f} MiB/s\n")

    lines.append("=== Reconstruction quality (plain vs decrypted) ===")
    lines.append(f"MAE:  {MAE}")
    lines.append(f"PSNR: {PSNR}")
    lines.append(f"SSIM: {SSIM}\n")

    lines.append("=== Cipher statistics ON cipher_uint8.png (view derived from Cu32) ===")
    lines.append(f"cipher_uint8 shape: {cipher_u8.shape}")
    lines.append(f"Entropy: {ENT:.6f} bits")
    lines.append(f"AdjCorr-H: {CH:.6f}")
    lines.append(f"AdjCorr-V: {CV:.6f}")
    lines.append(f"AdjCorr-D: {CD:.6f}\n")

    lines.append("=== Differential test (flip 1 plaintext pixel) ON cipher_uint8 view ===")
    lines.append(f"NPCR: {NPCR:.6f}%")
    lines.append(f"UACI: {UACI:.6f}%\n")

    lines.append("=== Key sensitivity test (flip 1 key bit) ON cipher_uint8 view ===")
    lines.append(f"Key original: {KEY}")
    lines.append(f"Key flipped:  {KEY_FLIP}")
    lines.append(f"NPCR_key: {NPCR_key:.6f}%")
    lines.append(f"UACI_key: {UACI_key:.6f}%\n")

    lines.append("=== Wrong-key decryption quality (from cipher_rgba.png) ===")
    lines.append(f"Wrong key: {WRONG_KEY}")
    lines.append(f"MAE_wrong:  {MAE_wrong}")
    lines.append(f"PSNR_wrong: {PSNR_wrong}")
    lines.append(f"SSIM_wrong: {SSIM_wrong}\n")

    lines.append("=== Saved outputs ===")
    lines.append(os.path.abspath(args.out))
    lines.append("Images:")
    lines.append(" - plain.png")
    lines.append(" - cipher_rgba.png (REAL invertible ciphertext image)")
    lines.append(" - cipher_uint8.png (non-invertible view for stats)")
    lines.append(" - decrypted.png")
    lines.append(" - abs_diff.png")
    lines.append(" - decrypted_wrong_key.png")
    lines.append(" - cipher_uint8_keyflip.png")
    lines.append(" - cipher_diff_abs.png")
    lines.append("Plots:")
    lines.append(" - hist_plain.png / hist_cipher.png / hist_decrypted.png")
    lines.append(" - corr_cipher_h.png / corr_cipher_v.png / corr_cipher_d.png")
    lines.append("Data:")
    lines.append(" - cipher_u32.npy (REAL ciphertext)")
    lines.append(" - cipher_D.npy   (optional float intermediate)")
    lines.append("State:")
    lines.append(" - state.npz (needed for decrypt_only robustness tests)")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.out, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
