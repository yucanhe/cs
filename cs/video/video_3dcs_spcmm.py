#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import time
import argparse
import hashlib
import numpy as np

import imageio.v2 as imageio
from scipy.fftpack import dct, idct

U32_MASK = np.uint64(0xFFFFFFFF)


# =========================================================
# Utilities
# =========================================================

def u32(x: int) -> np.uint32:
    return np.uint32(np.uint64(x) & U32_MASK)

def u32_arr(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.uint64) & U32_MASK).astype(np.uint32)

def sha256_bytes(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def now():
    return time.time()

def safe_macroblock_writer(out_file: str, fps: float):
    # avoid imageio auto-resize to 16 multiples
    return imageio.get_writer(out_file, fps=fps, macro_block_size=1)

def to_gray_u8(frame_rgb: np.ndarray) -> np.ndarray:
    if frame_rgb.ndim == 2:
        g = frame_rgb
    else:
        g = np.mean(frame_rgb.astype(np.float32), axis=2)
    return np.clip(g, 0, 255).astype(np.uint8)

def clip_i16(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -32768, 32767).astype(np.int16)


# =========================================================
# 3D-SPCMM chaos generator (deterministic)
# =========================================================

def chaos_seed(master_key: str, *tags: str) -> np.float64:
    b = master_key.encode("utf-8") + b"|" + "|".join(tags).encode("utf-8")
    h = sha256_bytes(b)
    v = int.from_bytes(h[:8], "big") & ((1 << 53) - 1)
    x = (v + 1) / (2**53 + 2)
    return np.float64(x)

def spcmm3_stream(master_key: str, cube_id: int, block_id: int, purpose: str):
    s0 = chaos_seed(master_key, f"cube={cube_id}", f"blk={block_id}", purpose, "x")
    s1 = chaos_seed(master_key, f"cube={cube_id}", f"blk={block_id}", purpose, "y")
    s2 = chaos_seed(master_key, f"cube={cube_id}", f"blk={block_id}", purpose, "z")

    x, y, z = float(s0), float(s1), float(s2)

    a = 3.99
    k = 0.13

    for _ in range(64):
        x = a * x * (1.0 - x)
        y = a * y * (1.0 - y)
        z = a * z * (1.0 - z)
        x = (x + k * (y - z)) % 1.0
        y = (y + k * (z - x)) % 1.0
        z = (z + k * (x - y)) % 1.0

    while True:
        x = a * x * (1.0 - x)
        y = a * y * (1.0 - y)
        z = a * z * (1.0 - z)
        x = (x + k * (y - z)) % 1.0
        y = (y + k * (z - x)) % 1.0
        z = (z + k * (x - y)) % 1.0
        v = (x + 0.37*y + 0.19*z) % 1.0
        yield v

def chaos_u32_stream(master_key: str, cube_id: int, block_id: int, purpose: str):
    g = spcmm3_stream(master_key, cube_id, block_id, purpose)
    while True:
        v = next(g)
        yield np.uint32(np.uint64(int(v * (2**32))) & U32_MASK)

def chaos_perm(master_key: str, cube_id: int, n: int, purpose: str):
    g = spcmm3_stream(master_key, cube_id, 0, purpose)
    vals = np.array([next(g) for _ in range(n)], dtype=np.float64)
    return np.argsort(vals).astype(np.int32)


# =========================================================
# S-Box from chaos (256 permutation)
# =========================================================

def sbox_from_chaos(master_key: str, cube_id: int):
    g = spcmm3_stream(master_key, cube_id, 0, "sbox")
    vals = np.array([next(g) for _ in range(256)], dtype=np.float64)
    idx = np.argsort(vals)
    sbox = np.arange(256, dtype=np.uint8)[idx]
    inv = np.zeros(256, dtype=np.uint8)
    inv[sbox] = np.arange(256, dtype=np.uint8)
    return sbox, inv


# =========================================================
# ARX-CBC vectorized (uint32)
# =========================================================

def arx_cbc_encrypt_u32(p: np.ndarray, ks: np.ndarray, prev_tag: np.uint32):
    assert p.dtype == np.uint32 and ks.dtype == np.uint32
    base = (p.astype(np.uint64) + ks.astype(np.uint64)) & U32_MASK
    c64 = (np.uint64(prev_tag) + np.add.accumulate(base, dtype=np.uint64)) & U32_MASK
    c = c64.astype(np.uint32)
    return c, c[-1]

def arx_cbc_decrypt_u32(c: np.ndarray, ks: np.ndarray, prev_tag: np.uint32):
    assert c.dtype == np.uint32 and ks.dtype == np.uint32
    prevs = np.empty_like(c)
    prevs[0] = prev_tag
    prevs[1:] = c[:-1]
    p64 = (c.astype(np.uint64) - ks.astype(np.uint64) - prevs.astype(np.uint64)) & U32_MASK
    return p64.astype(np.uint32)

def sbox_apply_u32_words(x_u32: np.ndarray, sbox: np.ndarray):
    b = x_u32.view(np.uint8)
    b2 = sbox[b]
    return b2.view(np.uint32)

def inv_sbox_apply_u32_words(x_u32: np.ndarray, inv: np.ndarray):
    b = x_u32.view(np.uint8)
    b2 = inv[b]
    return b2.view(np.uint32)


# =========================================================
# 3D DCT / IDCT (orthonormal)
# =========================================================

def dct3(x):
    y = dct(x, axis=0, norm="ortho")
    y = dct(y, axis=1, norm="ortho")
    y = dct(y, axis=2, norm="ortho")
    return y

def idct3(x):
    y = idct(x, axis=0, norm="ortho")
    y = idct(y, axis=1, norm="ortho")
    y = idct(y, axis=2, norm="ortho")
    return y


# =========================================================
# Measurement Matrix Phi (generated, not stored)
# =========================================================

def phi_gaussian(master_key: str, cube_id: int, block_id: int, m: int, n: int):
    g = spcmm3_stream(master_key, cube_id, block_id, f"phi_m={m}_n={n}")
    vals = np.empty(m * n, dtype=np.float32)
    i = 0
    while i < m * n:
        u1 = max(next(g), 1e-12)
        u2 = next(g)
        r = math.sqrt(-2.0 * math.log(u1))
        z0 = r * math.cos(2 * math.pi * u2)
        vals[i] = np.float32(z0)
        i += 1
    Phi = vals.reshape(m, n)
    denom = np.sqrt(np.sum(Phi * Phi, axis=1, keepdims=True)) + 1e-8
    Phi = Phi / denom
    return Phi.astype(np.float32)


def phi_bernoulli(master_key: str, cube_id: int, block_id: int, m: int, n: int):
    """
    混沌Bernoulli测量矩阵 (±1)
    比高斯矩阵更快，且满足RIP性质
    """
    g = spcmm3_stream(master_key, cube_id, block_id, f"phi_bern_m={m}_n={n}")
    vals = np.empty(m * n, dtype=np.float32)
    i = 0
    while i < m * n:
        u = next(g)
        vals[i] = np.float32(1.0 if u > 0.5 else -1.0)
        i += 1
    Phi = vals.reshape(m, n)
    Phi = Phi / math.sqrt(m)
    return Phi.astype(np.float32)


def phi_scrambled_fft(master_key: str, cube_id: int, block_id: int, m: int, n: int):
    """
    混沌Scrambled FFT测量矩阵
    """
    g = spcmm3_stream(master_key, cube_id, block_id, f"phi_fft_m={m}_n={n}")
    phases = np.array([next(g) * 2 * np.pi for _ in range(n)], dtype=np.float32)
    fft_matrix = np.fft.fft(np.eye(n)) / np.sqrt(n)
    phase_diag = np.diag(np.exp(1j * phases))
    scrambled = np.real(fft_matrix @ phase_diag)
    Phi = scrambled[:m, :].astype(np.float32)
    norms = np.sqrt(np.sum(Phi ** 2, axis=1, keepdims=True)) + 1e-8
    Phi = Phi / norms
    return Phi


def get_measurement_matrix(method: str = "gaussian"):
    """工厂函数：获取测量矩阵生成器"""
    methods = {"gaussian": phi_gaussian, "bernoulli": phi_bernoulli, "fft": phi_scrambled_fft}
    return methods.get(method, phi_gaussian)


# =========================================================
# FISTA for LASSO (Hot-start)
# =========================================================

def power_iter_spectral_norm_sq(A, iters=8):
    m, n = A.shape
    v = np.random.RandomState(123).randn(n).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-12)
    for _ in range(iters):
        Av = A @ v
        AtAv = A.T @ Av
        nv = np.linalg.norm(AtAv) + 1e-12
        v = (AtAv / nv).astype(np.float32)
    Av = A @ v
    return float(np.dot(Av, Av))

def soft_threshold(x, th):
    return np.sign(x) * np.maximum(np.abs(x) - th, 0.0)

def fista_l1(Phi, y, lam=0.002, iters=40, x0=None, L=None):
    m, n = Phi.shape
    if x0 is None:
        x = np.zeros(n, dtype=np.float32)
    else:
        x = x0.astype(np.float32).copy()

    z = x.copy()
    t = 1.0

    if L is None:
        L = power_iter_spectral_norm_sq(Phi, iters=6) + 1e-6

    PhiT = Phi.T
    step = 1.0 / L
    th = lam * step

    for _ in range(iters):
        grad = PhiT @ (Phi @ z - y)
        x_new = soft_threshold(z - step * grad, th)
        t_new = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x = x_new
        t = t_new

    return x, L


# =========================================================
# Block helpers
# =========================================================

def split_blocks(frames_u8: np.ndarray, B: int):
    T, H, W = frames_u8.shape
    Hb = (H // B) * B
    Wb = (W // B) * B
    frames = frames_u8[:, :Hb, :Wb]

    bh = Hb // B
    bw = Wb // B
    nb = bh * bw

    blocks = np.empty((nb, B, B, T), dtype=np.float32)
    idx = 0
    for i in range(bh):
        for j in range(bw):
            patch = frames[:, i*B:(i+1)*B, j*B:(j+1)*B].astype(np.float32)
            blocks[idx] = np.transpose(patch, (1,2,0))
            idx += 1

    return blocks, (H, W, Hb, Wb, bh, bw)

def merge_blocks(blocks: np.ndarray, meta, B: int, T: int):
    H, W, Hb, Wb, bh, bw = meta
    frames = np.zeros((T, Hb, Wb), dtype=np.float32)
    idx = 0
    for i in range(bh):
        for j in range(bw):
            cube = blocks[idx]
            patch = np.transpose(cube, (2,0,1))
            frames[:, i*B:(i+1)*B, j*B:(j+1)*B] = patch
            idx += 1

    frames_u8 = np.clip(frames, 0, 255).astype(np.uint8)

    out = np.zeros((T, H, W), dtype=np.uint8)
    out[:, :Hb, :Wb] = frames_u8
    if Hb < H:
        out[:, Hb:, :Wb] = out[:, Hb-1: Hb, :Wb]
    if Wb < W:
        out[:, :Hb, Wb:] = out[:, :Hb, Wb-1: Wb]
    if Hb < H and Wb < W:
        out[:, Hb:, Wb:] = out[:, Hb-1: Hb, Wb-1: Wb]
    return out


# =========================================================
# Low-frequency selection
# =========================================================

def lowfreq_indices(B: int, T: int, lf_xy: int, lf_t: int):
    lf_xy = min(lf_xy, B)
    lf_t = min(lf_t, T)
    idxs = []
    for u in range(lf_xy):
        for v in range(lf_xy):
            for w in range(lf_t):
                idxs.append((u,v,w))
    return idxs

def pack_lowfreq(coeff: np.ndarray, idxs):
    return np.array([coeff[u,v,w] for (u,v,w) in idxs], dtype=np.float32)

def unpack_lowfreq(x: np.ndarray, B: int, T: int, idxs):
    coeff = np.zeros((B,B,T), dtype=np.float32)
    for k,(u,v,w) in enumerate(idxs):
        coeff[u,v,w] = x[k]
    return coeff


# =========================================================
# Quantization (fixed step, stable)
# =========================================================

def quantize_f32_to_i16(y: np.ndarray, q_step: float):
    """
    Uniform quant: y_q = round(y / q_step) to int16 (clipped).
    Dequant: y = y_q * q_step
    """
    y = y.astype(np.float32)
    q = np.round(y / float(q_step)).astype(np.int32)
    q = np.clip(q, -32768, 32767).astype(np.int16)
    return q

def dequantize_i16_to_f32(q: np.ndarray, q_step: float):
    return (q.astype(np.float32) * float(q_step)).astype(np.float32)

def quantize_i16_to_u32(x_i16: np.ndarray) -> np.ndarray:
    x = x_i16.astype(np.int32) + 32768
    return x.astype(np.uint32)

def dequantize_u32_to_i16(x_u32: np.ndarray) -> np.ndarray:
    x = x_u32.astype(np.int32) - 32768
    return clip_i16(x)


# =========================================================
# Cube Encrypt / Decrypt
# =========================================================

def encrypt_keycube_blocks(blocks: np.ndarray, master_key: str, cube_id: int,
                           prev_tag: np.uint32, idxs, m_rate_key: float,
                           lam: float, iters: int, hot_start: int,
                           perm_blocks: np.ndarray,
                           sbox: np.ndarray,
                           q_step: float,
                           use_arx: int,
                           use_sbox: int,
                           phi_method: str = "gaussian"):
    nb, B, _, T = blocks.shape
    n = len(idxs)

    blocks_scr = blocks[perm_blocks]
    enc_blocks = []

    # 获取测量矩阵生成函数
    if phi_method == "gaussian":
        phi_func = phi_gaussian
    elif phi_method == "fft":
        phi_func = phi_scrambled_fft
    elif phi_method == "circulant":
        phi_func = chaos_partial_circulant
    else:
        phi_func = phi_gaussian

    for bi in range(nb):
        coeff = dct3(blocks_scr[bi])
        x = pack_lowfreq(coeff, idxs)

        m = max(1, int(round(m_rate_key * n)))
        Phi = phi_func(master_key, cube_id, int(bi), m, n)
        y = (Phi @ x).astype(np.float32)

        # quantize to int16, then pack to uint32 for (optional) ARX/S-Box
        y_q_i16 = quantize_f32_to_i16(y, q_step=q_step)
        y_u32 = quantize_i16_to_u32(y_q_i16)

        if use_arx:
            ks_gen = chaos_u32_stream(master_key, cube_id, int(bi), "arx_keycube")
            ks = np.array([next(ks_gen) for _ in range(m)], dtype=np.uint32)
            c_u32, prev_tag = arx_cbc_encrypt_u32(y_u32, ks, prev_tag)
        else:
            c_u32 = y_u32
            # prev_tag unchanged

        if use_sbox:
            c_u32 = sbox_apply_u32_words(c_u32, sbox)

        enc_blocks.append({
            "m": int(m),
            "q_step": float(q_step),
            "ct_u32": c_u32,
            "use_arx": int(use_arx),
            "use_sbox": int(use_sbox),
        })

    pkg = {
        "mode": "key",
        "cube_id": int(cube_id),
        "perm": perm_blocks.astype(np.int32),
        "enc_blocks": enc_blocks,
        "nb": int(nb),
    }
    return pkg, prev_tag


def decrypt_keycube_blocks(pkg, master_key: str, prev_tag: np.uint32,
                           idxs, B: int, T: int,
                           lam: float, iters: int, hot_start: int,
                           inv_sbox: np.ndarray,
                           use_arx: int,
                           use_sbox: int,
                           phi_method: str = "gaussian"):
    nb = int(pkg["nb"])
    perm = pkg["perm"].astype(np.int32)
    inv_perm = np.argsort(perm)
    blocks_scr = np.empty((nb, B, B, T), dtype=np.float32)

    x0_cache = None
    L_cache = None

    # 获取测量矩阵生成函数
    if phi_method == "gaussian":
        phi_func = phi_gaussian
    elif phi_method == "fft":
        phi_func = phi_scrambled_fft
    elif phi_method == "circulant":
        phi_func = chaos_partial_circulant
    else:
        phi_func = phi_gaussian

    for bi in range(nb):
        enc = pkg["enc_blocks"][bi]
        m = int(enc["m"])
        q_step = float(enc.get("q_step", 0.01))
        c_u32 = enc["ct_u32"].astype(np.uint32)

        if use_sbox:
            c_u32 = inv_sbox_apply_u32_words(c_u32, inv_sbox)

        if use_arx:
            ks_gen = chaos_u32_stream(master_key, int(pkg["cube_id"]), int(bi), "arx_keycube")
            ks = np.array([next(ks_gen) for _ in range(m)], dtype=np.uint32)
            y_u32 = arx_cbc_decrypt_u32(c_u32, ks, prev_tag)
            prev_tag = c_u32[-1]
        else:
            y_u32 = c_u32
            # prev_tag unchanged

        # unpack quantized measurements
        y_q_i16 = dequantize_u32_to_i16(y_u32)
        y = dequantize_i16_to_f32(y_q_i16, q_step=q_step)

        n = len(idxs)
        Phi = phi_func(master_key, int(pkg["cube_id"]), int(bi), m, n)

        if hot_start and (x0_cache is not None):
            x0 = x0_cache
            L = L_cache
        else:
            x0 = None
            L = None

        x_rec, L_new = fista_l1(Phi, y, lam=lam, iters=iters, x0=x0, L=L)
        x0_cache = x_rec
        L_cache = L_new

        coeff = unpack_lowfreq(x_rec, B, T, idxs)
        block = idct3(coeff)
        blocks_scr[bi] = block

    blocks = blocks_scr[inv_perm]
    return blocks, prev_tag


def encrypt_nonkey_residual_blocks(blocks: np.ndarray, prev_key_blocks: np.ndarray,
                                  master_key: str, cube_id: int, prev_tag: np.uint32,
                                  perm_blocks: np.ndarray, sbox: np.ndarray,
                                  keep_rate: float,
                                  use_arx: int,
                                  use_sbox: int):
    nb, B, _, T = blocks.shape
    blocks_scr = blocks[perm_blocks]
    prev_scr = prev_key_blocks[perm_blocks]

    enc_blocks = []
    for bi in range(nb):
        res = blocks_scr[bi] - prev_scr[bi]
        res_i16 = clip_i16(np.round(res))

        if keep_rate < 1.0:
            n_all = res_i16.size
            keep = max(1, int(round(keep_rate * n_all)))
            perm = chaos_perm(master_key, cube_id * 131 + bi, n_all, "res_keep")
            idx_keep = perm[:keep]
            data = res_i16.reshape(-1)[idx_keep]
            mode = "packed"
        else:
            idx_keep = None
            data = res_i16.reshape(-1)
            mode = "full"

        p_u32 = quantize_i16_to_u32(data)

        if use_arx:
            ks_gen = chaos_u32_stream(master_key, cube_id, int(bi), "arx_residual")
            ks = np.array([next(ks_gen) for _ in range(len(p_u32))], dtype=np.uint32)
            c_u32, prev_tag = arx_cbc_encrypt_u32(p_u32, ks, prev_tag)
        else:
            c_u32 = p_u32

        if use_sbox:
            c_u32 = sbox_apply_u32_words(c_u32, sbox)

        enc_blocks.append({
            "mode": mode,
            "idx_keep": idx_keep.astype(np.int32) if idx_keep is not None else None,
            "n_all": int(res_i16.size),
            "ct_u32": c_u32,
        })

    pkg = {
        "mode": "nonkey",
        "cube_id": int(cube_id),
        "perm": perm_blocks.astype(np.int32),
        "enc_blocks": enc_blocks,
        "nb": int(nb),
        "B": int(B),
        "T": int(T),
    }
    return pkg, prev_tag


def decrypt_nonkey_residual_blocks(pkg, prev_key_blocks: np.ndarray,
                                  master_key: str, prev_tag: np.uint32,
                                  inv_sbox: np.ndarray,
                                  use_arx: int,
                                  use_sbox: int):
    nb = int(pkg["nb"])
    perm = pkg["perm"].astype(np.int32)
    inv_perm = np.argsort(perm)

    B = int(pkg["B"])
    T = int(pkg["T"])

    prev_scr = prev_key_blocks[perm]
    out_scr = np.empty((nb, B, B, T), dtype=np.float32)

    for bi in range(nb):
        enc = pkg["enc_blocks"][bi]
        c_u32 = enc["ct_u32"].astype(np.uint32)

        if use_sbox:
            c_u32 = inv_sbox_apply_u32_words(c_u32, inv_sbox)

        if use_arx:
            ks_gen = chaos_u32_stream(master_key, int(pkg["cube_id"]), int(bi), "arx_residual")
            ks = np.array([next(ks_gen) for _ in range(len(c_u32))], dtype=np.uint32)
            p_u32 = arx_cbc_decrypt_u32(c_u32, ks, prev_tag)
            prev_tag = c_u32[-1]
        else:
            p_u32 = c_u32

        data_i16 = dequantize_u32_to_i16(p_u32)

        if enc["mode"] == "full":
            res_flat = data_i16.astype(np.int16)
        else:
            n_all = int(enc["n_all"])
            idx_keep = enc["idx_keep"].astype(np.int32)
            res_flat = np.zeros(n_all, dtype=np.int16)
            res_flat[idx_keep] = data_i16

        res = res_flat.reshape(B, B, T).astype(np.float32)
        out_scr[bi] = prev_scr[bi] + res

    out_blocks = out_scr[inv_perm]
    return out_blocks, prev_tag


# =========================================================
# Streaming cube iterator
# =========================================================

def iter_cubes(reader, cube_t: int, chunk_frames: int):
    buf = []
    for frame in reader:
        buf.append(to_gray_u8(frame))
        if len(buf) >= chunk_frames:
            while len(buf) >= cube_t:
                cube = np.stack(buf[:cube_t], axis=0)
                buf = buf[cube_t:]
                yield cube
    while len(buf) >= cube_t:
        cube = np.stack(buf[:cube_t], axis=0)
        buf = buf[cube_t:]
        yield cube


# =========================================================
# Encrypt / Decrypt full video
# =========================================================

def encrypt_video(in_file: str, out_manifest: str, master_key: str,
                  chunk_frames: int, cube_t: int, block: int,
                  key_stride: int,
                  m_rate_key: float,
                  lf_xy: int, lf_t: int,
                  q_step: float,
                  lam: float, iters: int, hot_start: int,
                  use_arx: int, use_sbox: int,
                  keep_nonkey_y: float = 1.0,
                  phi_method: str = "gaussian"):
    t0 = now()
    reader = imageio.get_reader(in_file)
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 30.0))

    first = reader.get_next_data()
    H, W = to_gray_u8(first).shape
    reader.close()
    reader = imageio.get_reader(in_file)

    chunks_dir = os.path.splitext(out_manifest)[0] + "_chunks"
    ensure_dir(chunks_dir)

    idxs = lowfreq_indices(block, cube_t, lf_xy, lf_t)
    prev_tag = np.uint32(0x12345678)

    cube_id = 0
    total_cubes = 0
    prev_key_blocks = None
    stored_files = []

    print("=== Encrypt (Selective 3D-DCT+CS + Quant(i16) + optional ARX/S-Box) ===")
    print(f"Input: {in_file}")
    print(f"Output manifest: {out_manifest}")
    print(f"Chunks dir: {chunks_dir}")
    print(f"fps={fps:.3f} size={W}x{H} cube_t={cube_t} block={block} chunk={chunk_frames}")
    print(f"key_stride={key_stride} m_rate_key={m_rate_key} lf=(xy={lf_xy},t={lf_t}) q_step={q_step}")
    print(f"arx={use_arx} sbox={use_sbox} keep_nonkey_y={keep_nonkey_y}")
    print(f"phi_method={phi_method}")

    for cube in iter_cubes(reader, cube_t=cube_t, chunk_frames=chunk_frames):
        blocks, meta_blk = split_blocks(cube, block)
        nb = blocks.shape[0]

        sbox, _inv = sbox_from_chaos(master_key, cube_id=cube_id)
        perm_blocks = chaos_perm(master_key, cube_id, nb, "global_block_perm")

        is_key = (cube_id % max(1, key_stride) == 0) or (prev_key_blocks is None)

        if is_key:
            mode = "keycube_3ddct_cs"
            pkg, prev_tag = encrypt_keycube_blocks(
                blocks, master_key, cube_id, prev_tag, idxs,
                m_rate_key=m_rate_key, lam=lam, iters=iters, hot_start=hot_start,
                perm_blocks=perm_blocks, sbox=sbox,
                q_step=q_step,
                use_arx=use_arx, use_sbox=use_sbox,
                phi_method=phi_method
            )
            prev_key_blocks = blocks.copy()
        else:
            mode = "nonkey_residual"
            pkg, prev_tag = encrypt_nonkey_residual_blocks(
                blocks, prev_key_blocks, master_key, cube_id, prev_tag,
                perm_blocks=perm_blocks, sbox=sbox,
                keep_rate=keep_nonkey_y,
                use_arx=use_arx, use_sbox=use_sbox
            )

        pkg["mode_name"] = mode
        pkg["meta_blk"] = {
            "H": int(meta_blk[0]), "W": int(meta_blk[1]),
            "Hb": int(meta_blk[2]), "Wb": int(meta_blk[3]),
            "bh": int(meta_blk[4]), "bw": int(meta_blk[5]),
        }

        cube_file = os.path.join(chunks_dir, f"cube_{cube_id:06d}.npz")
        np.savez_compressed(cube_file, pkg=np.array([pkg], dtype=object))
        stored_files.append(os.path.basename(cube_file))

        total_cubes += 1
        if cube_id % 10 == 0:
            print(f"  [encrypt] cube {cube_id}  mode={mode}")

        cube_id += 1

    reader.close()

    manifest = {
        "version": 3,
        "in_file": os.path.basename(in_file),
        "fps": fps,
        "H": int(H),
        "W": int(W),
        "cube_t": int(cube_t),
        "block": int(block),
        "lf_xy": int(lf_xy),
        "lf_t": int(lf_t),
        "key_stride": int(key_stride),
        "m_rate_key": float(m_rate_key),
        "q_step": float(q_step),
        "lam": float(lam),
        "iters": int(iters),
        "hot_start": int(hot_start),
        "arx": int(use_arx),
        "sbox": int(use_sbox),
        "keep_nonkey_y": float(keep_nonkey_y),
        "phi_method": phi_method,
        "chunks_dir": os.path.basename(chunks_dir),
        "files": stored_files,
    }
    np.savez_compressed(out_manifest, manifest=json.dumps(manifest).encode("utf-8"))

    print("=== Encrypt Summary ===")
    print(f"cubes={total_cubes}  time={now()-t0:.3f}s")
    print(f"Saved manifest: {out_manifest}")


def decrypt_video(in_manifest: str, out_file: str, master_key: str,
                  lam: float, iters: int, hot_start: int,
                  use_arx_override: int = -1,
                  use_sbox_override: int = -1):
    t0 = now()
    man_npz = np.load(in_manifest, allow_pickle=True)
    manifest = json.loads(man_npz["manifest"].tobytes().decode("utf-8"))

    chunks_dir = os.path.join(os.path.dirname(in_manifest), manifest["chunks_dir"])
    files = manifest["files"]

    fps = float(manifest["fps"])
    H = int(manifest["H"])
    W = int(manifest["W"])
    cube_t = int(manifest["cube_t"])
    block = int(manifest["block"])
    lf_xy = int(manifest["lf_xy"])
    lf_t = int(manifest["lf_t"])

    use_arx = int(manifest.get("arx", 1)) if use_arx_override < 0 else int(use_arx_override)
    use_sbox = int(manifest.get("sbox", 1)) if use_sbox_override < 0 else int(use_sbox_override)
    phi_method = manifest.get("phi_method", "gaussian")

    idxs = lowfreq_indices(block, cube_t, lf_xy, lf_t)
    prev_tag = np.uint32(0x12345678)
    prev_key_blocks = None

    print("=== Decrypt (Selective 3D-DCT+CS + Quant(i16) + optional ARX/S-Box) ===")
    print(f"Manifest: {in_manifest}")
    print(f"Chunks: {chunks_dir}")
    print(f"Output: {out_file}")
    print(f"fps={fps} cubes={len(files)} arx={use_arx} sbox={use_sbox}")

    writer = safe_macroblock_writer(out_file, fps=fps)
    written_frames = 0

    for ci, fname in enumerate(files):
        cube_path = os.path.join(chunks_dir, fname)
        z = np.load(cube_path, allow_pickle=True)
        pkg = z["pkg"][0]
        cube_id = int(pkg["cube_id"])

        _sbox, inv_sbox = sbox_from_chaos(master_key, cube_id=cube_id)

        meta_blk = pkg["meta_blk"]
        bh = int(meta_blk["bh"]); bw = int(meta_blk["bw"])
        nb = bh * bw

        mode = pkg["mode"]
        mode_name = pkg.get("mode_name", mode)

        if mode == "key":
            blocks, prev_tag = decrypt_keycube_blocks(
                pkg, master_key, prev_tag, idxs,
                B=block, T=cube_t, lam=lam, iters=iters, hot_start=hot_start,
                inv_sbox=inv_sbox,
                use_arx=use_arx, use_sbox=use_sbox,
                phi_method=phi_method
            )
            prev_key_blocks = blocks.copy()
        else:
            if prev_key_blocks is None:
                raise RuntimeError("Non-key cube encountered before any key cube. key_stride too large?")
            blocks, prev_tag = decrypt_nonkey_residual_blocks(
                pkg, prev_key_blocks, master_key, prev_tag,
                inv_sbox=inv_sbox,
                use_arx=use_arx, use_sbox=use_sbox
            )

        frames_u8 = merge_blocks(
            blocks,
            (H, W, (H//block)*block, (W//block)*block, bh, bw),
            block, cube_t
        )

        for t in range(cube_t):
            g = frames_u8[t]
            rgb = np.stack([g, g, g], axis=2)
            writer.append_data(rgb)
            written_frames += 1

        if ci % 10 == 0:
            print(f"  [decrypt] cube {ci}/{len(files)}  mode={mode_name}  written={written_frames}")

    writer.close()
    print("=== Decrypt Summary ===")
    print(f"frames={written_frames}  time={now()-t0:.3f}s")
    print(f"Saved: {out_file}")


# =========================================================
# CLI
# =========================================================

def build_argparser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("encrypt")
    pe.add_argument("--in_file", required=True)
    pe.add_argument("--out_file", required=True, help="manifest npz")
    pe.add_argument("--key", required=True)

    pe.add_argument("--chunk", type=int, default=60, help="frames buffer size (streaming)")
    pe.add_argument("--cube_t", type=int, default=8, help="time length per cube")
    pe.add_argument("--block", type=int, default=16, help="spatial block size")

    pe.add_argument("--key_stride", type=int, default=2, help="every N cubes is key cube")
    pe.add_argument("--m_rate_key", type=float, default=0.7)

    pe.add_argument("--lf_xy", type=int, default=10)
    pe.add_argument("--lf_t", type=int, default=6)

    pe.add_argument("--q_step", type=float, default=0.015, help="uniform quant step for y")
    pe.add_argument("--lam", type=float, default=0.0015)
    pe.add_argument("--iters", type=int, default=140)
    pe.add_argument("--hot_start", type=int, default=1)

    # NEW: toggles
    pe.add_argument("--arx", type=int, default=1, choices=[0, 1], help="enable ARX-CBC")
    pe.add_argument("--sbox", type=int, default=1, choices=[0, 1], help="enable S-Box")

    # non-key residual keep rate (acts like compress for nonkey)
    pe.add_argument("--keep_nonkey_y", type=float, default=1.0)

    # Measurement matrix type (chaos-based only)
    pe.add_argument("--phi_method", type=str, default="gaussian", 
                   choices=["gaussian", "bernoulli", "fft", "circulant"],
                   help="chaos-based measurement matrix: gaussian, bernoulli, fft, or circulant")

    pd = sub.add_parser("decrypt")
    pd.add_argument("--in_file", required=True, help="manifest npz")
    pd.add_argument("--out_file", required=True, help="output mp4")
    pd.add_argument("--key", required=True)
    pd.add_argument("--lam", type=float, default=0.0015)
    pd.add_argument("--iters", type=int, default=140)
    pd.add_argument("--hot_start", type=int, default=1)

    # NEW: allow override; -1 means follow manifest
    pd.add_argument("--arx", type=int, default=-1, choices=[-1, 0, 1], help="override ARX (default follow manifest)")
    pd.add_argument("--sbox", type=int, default=-1, choices=[-1, 0, 1], help="override S-Box (default follow manifest)")

    return p


def main():
    args = build_argparser().parse_args()
    if args.cmd == "encrypt":
        encrypt_video(
            in_file=args.in_file,
            out_manifest=args.out_file,
            master_key=args.key,
            chunk_frames=args.chunk,
            cube_t=args.cube_t,
            block=args.block,
            key_stride=args.key_stride,
            m_rate_key=args.m_rate_key,
            lf_xy=args.lf_xy,
            lf_t=args.lf_t,
            q_step=args.q_step,
            lam=args.lam,
            iters=args.iters,
            hot_start=args.hot_start,
            use_arx=args.arx,
            use_sbox=args.sbox,
            keep_nonkey_y=args.keep_nonkey_y,
            phi_method=args.phi_method
        )
    else:
        decrypt_video(
            in_manifest=args.in_file,
            out_file=args.out_file,
            master_key=args.key,
            lam=args.lam,
            iters=args.iters,
            hot_start=args.hot_start,
            use_arx_override=args.arx,
            use_sbox_override=args.sbox
        )


if __name__ == "__main__":
    main()
