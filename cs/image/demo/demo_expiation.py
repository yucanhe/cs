"""

我保留你原有结构、函数名与流程，只在关键位置加入详细中文注释。

用途：把“压缩感知测量 + 测量域加密 + 可逆密文图封装 + 解密重建(FISTA)”完整跑通，
并输出论文常用安全/质量指标与图像结果。

运行示例：
python demo_fista_annotated.py --img 1.png --cr 0.5 --lam 0.01 --iter 150
"""

import argparse
import hashlib
import os
import time
import numpy as np
from dataclasses import dataclass
from scipy.fftpack import dct, idct
from PIL import Image
import matplotlib.pyplot as plt

# SSIM：优先用 skimage（更标准），没有则用简单自实现
try:
    from skimage.metrics import structural_similarity as ssim_skimage
except Exception:
    ssim_skimage = None

# -----------------------
# Optional numba
# -----------------------
# Numba 用来加速三重循环（3D 扩散 + uint32 ARX 扩散等）
NUMBA_OK = True
try:
    from numba import njit
except Exception:
    NUMBA_OK = False
    def njit(*args, **kwargs):
        def deco(fn): return fn
        return deco

# uint32 运算的“模”与掩码
UINT32_MOD = 2**32
MOD_MASK = 0xFFFFFFFF


# ============================================================
# Basic utils
# ============================================================

def ensure_dir(path: str):
    """确保输出目录存在"""
    os.makedirs(path, exist_ok=True)

def frac_arr(x: np.ndarray) -> np.ndarray:
    """
    取小数部分：x - floor(x)
    作用：把数据强制落入 [0,1) 区间，避免扩散递推数值发散
    """
    return x - np.floor(x)

def dct2(a: np.ndarray) -> np.ndarray:
    """二维 DCT（正交归一）"""
    return dct(dct(a.T, norm="ortho").T, norm="ortho")

def idct2(a: np.ndarray) -> np.ndarray:
    """二维 IDCT（正交归一）"""
    return idct(idct(a.T, norm="ortho").T, norm="ortho")

def blockify(img: np.ndarray, B: int = 8):
    """
    图像分块：H×W -> N×B×B
    返回 blocks, (gh, gw)
    gh = H/B 块行数；gw = W/B 块列数；N = gh*gw
    """
    H, W = img.shape
    assert H % B == 0 and W % B == 0, "H,W must be divisible by B"
    blocks = []
    for r in range(0, H, B):
        for c in range(0, W, B):
            blocks.append(img[r:r+B, c:c+B])
    return np.array(blocks), (H // B, W // B)

def unblockify(blocks: np.ndarray, grid_shape, B: int = 8):
    """
    拼块：N×B×B -> H×W
    """
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
    """
    key -> 64-bit seed
    用 SHA256 前 8 字节作为随机种子，保证确定性。
    """
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)

def flip_one_bit_in_key(key: str) -> str:
    """
    把 key 的第一个字节 xor 1，用于“密钥敏感性”测试。
    """
    b = bytearray(key.encode("utf-8"))
    if len(b) == 0:
        b = bytearray(b"\x00")
    b[0] ^= 0x01
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.hex()


# ============================================================
# Python-safe rotations (CRITICAL for non-numba correctness)
# ============================================================

def rotl32_py(x: int, r: int) -> int:
    """
    uint32 左旋：用于 ARX 扩散
    Python int 无限精度，因此必须 & 0xFFFFFFFF 固定在 32 位
    """
    r &= 31
    x &= MOD_MASK
    return ((x << r) | (x >> (32 - r))) & MOD_MASK

def rotr32_py(x: int, r: int) -> int:
    """uint32 右旋"""
    r &= 31
    x &= MOD_MASK
    return ((x >> r) | (x << (32 - r))) & MOD_MASK


# ============================================================
# Chaos generator (demo SPCMM-like; deterministic)
# ============================================================

def spcmm_generate_xyz(key: str, n: int, burn: int = 2000):
    """
    生成三路 [0,1) 的“超混沌”序列 x,y,z（确定性版本）
    - seed 从 key 来
    - logistic + 正弦耦合（不是你论文真正 3D-SPCMM，但用于工程演示足够）
    - burn-in 丢弃前 burn 项，减少初值影响
    """
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
    """
    为第 i 个块生成“唯一的”测量矩阵 Phi_i (m_meas x n)
    - msg 融合 key + z_val + block_id，让每块不同且可复现
    - 默认用高斯随机矩阵并行归一化（常见 CS sensing matrix）
    - 特殊：m=n 时做 QR 得到正交矩阵（数值更稳，且便于直接 Phi^T 近似逆）
    """
    msg = f"{key}|{z_val:.16f}|{block_id}|{m_meas}|{n}"
    seed = _hash_to_seed(msg)
    rng = np.random.default_rng(seed)

    A = rng.standard_normal((m_meas, n)).astype(np.float64)

    if m_meas == n:
        # QR 得到正交矩阵 Q（行列都正交，能量保持）
        Q, R = np.linalg.qr(A)
        s = np.sign(np.diag(R))
        s[s == 0] = 1.0
        Q = Q * s
        return Q.astype(np.float64)

    # 行归一化，使每行单位范数（有利于重建稳定）
    A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    return A


# ============================================================
# Float 3D diffusion (numba optional)
# ============================================================

@njit(cache=True)
def diffusion3d_forward_numba(Yn, mask, a, b, c):
    """
    浮点 3D 扩散（加密方向）
    D[i,j,k] 依赖：
      - up   = D[i-1,j,k]  (H轴方向)
      - left = D[i,j-1,k]  (W轴方向)
      - prev = D[i,j,k-1]  (M轴方向)
    然后把结果取 frac 保持在 [0,1)。
    """
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
    """
    浮点 3D 扩散的逆（解密方向）
    因为扩散是“有向递推”，按相同扫描顺序即可逐点还原 Yn。
    """
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

def diffusion3d_forward(Yn, mask, a=0.21, b=0.33, c=0.27):
    """包装：有 numba 则走 numba，否则走 python"""
    if NUMBA_OK:
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

def diffusion3d_inverse(D, mask, a=0.21, b=0.33, c=0.27):
    """包装：逆扩散"""
    if NUMBA_OK:
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
    """
    把浮点 [0,1) 映射到 uint32：
      U = floor( frac(D) * 2^32 )
    这一步使“后续的密文扩散/封装”完全可逆（离散化）。
    """
    D = frac_arr(D.astype(np.float64))
    return np.floor(D * UINT32_MOD).astype(np.uint32)

def float01_from_u32(U: np.ndarray) -> np.ndarray:
    """uint32 -> float [0,1) 反量化"""
    return (U.astype(np.float64) / UINT32_MOD)

def u32_keystream(key: str, shape) -> np.ndarray:
    """
    生成 uint32 伪随机序列（mask/ks），由 key 决定，确保解密端可复现。
    """
    seed = _hash_to_seed(key)
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**32, size=np.prod(shape), dtype=np.uint32).reshape(shape)

@njit(cache=True)
def rotl32_numba(x, r):
    """numba 版左旋"""
    r &= 31
    return ((x << r) | (x >> (32 - r))) & MOD_MASK

@njit(cache=True)
def rotr32_numba(x, r):
    """numba 版右旋"""
    r &= 31
    return ((x >> r) | (x << (32 - r))) & MOD_MASK

@njit(cache=True)
def sub_bytes_u32_numba(x, sbox):
    """
    把 uint32 拆成 4 个字节，对每个字节查表替换，再拼回 uint32。
    这是非线性（类似 AES 的 SubBytes，但这里是动态 S-box）。
    """
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
    """python 版 SubBytes（用于不启用 numba 时）"""
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
    """
    为每个 depth k 生成一层 S-box（256置换），并同时生成逆 S-box。
    - sbox_layers[k]：把 byte 映射到 byte（置换）
    - inv_layers[k] ：逆置换
    注意：S-box 完全由 key 决定，解密端可复现，不需要传侧信息。
    """
    sbox_layers = np.empty((M, 256), dtype=np.uint8)
    inv_layers = np.empty((M, 256), dtype=np.uint8)
    for kk in range(M):
        xs, _, _ = spcmm_generate_xyz(f"{key}|{label}|depth={kk}", 256 + 64)
        seq = xs[:256]
        perm = np.argsort(seq).astype(np.uint8)  # argsort 结果是 0..255 的置换
        sbox_layers[kk] = perm
        inv = np.empty(256, dtype=np.uint8)
        inv[sbox_layers[kk]] = np.arange(256, dtype=np.uint8)
        inv_layers[kk] = inv
    return sbox_layers, inv_layers


# ============================================================
# uint32 ARX + dynamic S-box per depth (forward/inverse)
# ============================================================

@njit(cache=True)
def u32_pass1_forward_numba(U, mask1, ks1, sbox_layers):
    """
    uint32 扩散 pass1（正向扫描）
    C[i,j,k] 依赖上/左/前的 C（反馈扩散）
    核心结构：ADD + ROT + XOR + SubBytes（ARX + 非线性）
    """
    H, W, M = U.shape
    C = np.empty_like(U)
    for i in range(H):
        for j in range(W):
            for k in range(M):
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
def u32_pass2_backward_numba(C, mask2, ks2, sbox_layers):
    """
    uint32 扩散 pass2（反向扫描）
    这里依赖下/右/后，实现“全向扩散”效果。
    """
    H, W, M = C.shape
    for i in range(H - 1, -1, -1):
        for j in range(W - 1, -1, -1):
            for k in range(M - 1, -1, -1):
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
def u32_undo_pass2_forward_numba(C, mask2, ks2, inv_sbox_layers):
    """
    逆 pass2：必须按“正向扫描”来 undo（因为加密 pass2 是反向扫描）
    步骤：invSubBytes -> XOR -> rotr -> 减去邻居项与 mask2
    """
    H, W, M = C.shape
    C1 = C.copy()
    for i in range(H):
        for j in range(W):
            for k in range(M):
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
def u32_undo_pass1_backward_numba(C1, mask1, ks1, inv_sbox_layers):
    """
    逆 pass1：必须按“反向扫描” undo（因为加密 pass1 是正向扫描）
    """
    H, W, M = C1.shape
    U = np.empty_like(C1)
    for i in range(H - 1, -1, -1):
        for j in range(W - 1, -1, -1):
            for k in range(M - 1, -1, -1):
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

def u32_diffuse_forward(U: np.ndarray, key: str):
    """
    uint32 域的完整加密扩散：
    1) pass1 forward
    2) pass2 backward
    S-box 每个 depth 一层，mask/ks 都由 key 派生。
    """
    H, W, M = U.shape
    sbox1_layers, _ = build_sbox_layers_from_chaos(key, M, "sbox_pass1")
    sbox2_layers, _ = build_sbox_layers_from_chaos(key, M, "sbox_pass2")

    mask1 = u32_keystream(key + "|mask1|", U.shape)
    ks1   = u32_keystream(key + "|ks1|",   U.shape)
    mask2 = u32_keystream(key + "|mask2|", U.shape)
    ks2   = u32_keystream(key + "|ks2|",   U.shape)

    if NUMBA_OK:
        C = u32_pass1_forward_numba(U, mask1, ks1, sbox1_layers)
        C = u32_pass2_backward_numba(C, mask2, ks2, sbox2_layers)
        return C.astype(np.uint32)

    # python fallback（逻辑与 numba 相同）
    C = U.copy().astype(np.uint32)

    # pass1 forward
    for i in range(H):
        for j in range(W):
            for k in range(M):
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

def u32_diffuse_inverse(Cu32: np.ndarray, key: str):
    """
    uint32 扩散的完整逆：
    先 undo pass2（正向扫描）
    再 undo pass1（反向扫描）
    """
    H, W, M = Cu32.shape
    _, inv1_layers = build_sbox_layers_from_chaos(key, M, "sbox_pass1")
    _, inv2_layers = build_sbox_layers_from_chaos(key, M, "sbox_pass2")

    mask1 = u32_keystream(key + "|mask1|", Cu32.shape)
    ks1   = u32_keystream(key + "|ks1|",   Cu32.shape)
    mask2 = u32_keystream(key + "|mask2|", Cu32.shape)
    ks2   = u32_keystream(key + "|ks2|",   Cu32.shape)

    if NUMBA_OK:
        C1 = u32_undo_pass2_forward_numba(Cu32, mask2, ks2, inv2_layers)
        U  = u32_undo_pass1_backward_numba(C1,  mask1, ks1, inv1_layers)
        return U.astype(np.uint32)

    # python fallback（逻辑与 numba 相同）
    C1 = Cu32.copy().astype(np.uint32)

    # undo pass2 forward scan
    for i in range(H):
        for j in range(W):
            for k in range(M):
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
    """
    仅用于统计展示的 uint8 “视图”：
    取每个 uint32 的 4 个字节 xor 合成一个灰度值。
    重要：它不可逆！只是为了显示/算熵/相关/NPCR/UACI。
    """
    x = Cu32.astype(np.uint32)
    u8 = ((x >> 24) ^ (x >> 16) ^ (x >> 8) ^ x).astype(np.uint8)
    flat = u8.flatten()
    if flat.size == out_H * out_W:
        return flat.reshape((out_H, out_W))
    gh, gw, m = u8.shape
    return u8.reshape((gh, gw * m))

def save_cipher_rgba(Cu32: np.ndarray, out_path: str):
    """
    把真实密文 Cu32(gh,gw,m) 以 RGBA 图形式保存（可逆）：
    - 每个 uint32 -> 4 个字节 -> RGBA
    - 图像尺寸：(gh, gw*m)
    解密时读回 RGBA 再拼成 uint32，保证完全可逆。
    """
    H, W, M = Cu32.shape
    words = Cu32.reshape(-1).astype(np.uint32)
    b0 = ((words >> 24) & 0xFF).astype(np.uint8)
    b1 = ((words >> 16) & 0xFF).astype(np.uint8)
    b2 = ((words >> 8) & 0xFF).astype(np.uint8)
    b3 = (words & 0xFF).astype(np.uint8)
    rgba = np.stack([b0, b1, b2, b3], axis=1)
    img_rgba = rgba.reshape(H, W * M, 4)
    Image.fromarray(img_rgba, mode="RGBA").save(out_path)

def load_cipher_rgba_to_Cu32(path: str, H_: int, W_: int, M_: int) -> np.ndarray:
    """
    从 cipher_rgba.png 读回真实密文 Cu32（完全可逆）
    校验尺寸必须满足：(H_, W_*M_, 4)
    """
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
    """
    用幂迭代估计 Lipschitz 常数 L = ||Phi^T Phi||_2
    这是 FISTA 的步长关键：step = 1/L
    """
    n = Phi.shape[1]
    v = np.random.default_rng(0).standard_normal(n)
    v /= (np.linalg.norm(v) + 1e-12)
    for _ in range(iters):
        v = Phi.T @ (Phi @ v)
        v /= (np.linalg.norm(v) + 1e-12)
    w = Phi @ v
    return float(np.dot(w, w))

def soft_threshold(x: np.ndarray, lam: float):
    """L1 近端算子：soft-threshold"""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)

def fista_l1(Phi: np.ndarray, y: np.ndarray, lam: float, max_iter: int = 120, tol: float = 1e-5):
    """
    解 LASSO：
      min_x 0.5||Phi x - y||^2 + lam||x||_1
    输出 x（DCT系数向量）
    """
    n = Phi.shape[1]
    x = np.zeros(n, dtype=np.float64)
    z = x.copy()
    t = 1.0

    # 估计 L：越准确收敛越稳，但每块算一次会比较慢（可优化点）
    L = power_iteration_L(Phi, iters=10) + 1e-12
    invL = 1.0 / L

    prev = 1e30
    for _ in range(max_iter):
        # 梯度：Phi^T (Phi z - y)
        grad = Phi.T @ (Phi @ z - y)

        # 近端更新（soft threshold）
        x_new = soft_threshold(z - invL * grad, lam * invL)

        # Nesterov 加速（FISTA）
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        t = t_new
        x = x_new

        # 目标函数（用于早停）
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
    """
    解密必须用到的“侧信息”（但你不传输，脚本里直接保存/复用）
    真正论文里如果想不传，需要把这些都变成 key 可复现。
    目前：
    - y_min/y_max 用于反归一化（否则测量值尺度无法恢复）
    - diff_abc/mask_start/chaos_need 用于复现浮点扩散 mask
    """
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

class CSChaosCrypto:
    """
    封装“测量域加密/解密（浮点扩散部分）”
    注：uint32 ARX + S-box 是在 main 里做的。
    """
    def __init__(self, key: str, B=8, m_meas=64):
        self.key = key
        self.B = B
        self.n = 64
        self.m_meas = int(m_meas)

    def encrypt_prequant_D(self, img_u8: np.ndarray):
        """
        加密（到量化前 D）
        流程：
        1) Pn = P/255 归一化到 [0,1]
        2) 分块 -> DCT -> 得到每块 64 维 DCT系数 Omega_i
        3) 动态测量：y_i = Phi_i @ Omega_i  (Phi_i 每块不同)
        4) 堆叠成 3D 张量 Y(gh,gw,m)
        5) 归一化到 [0,1) 得 Yn
        6) 用混沌序列做 mask
        7) 3D 浮点扩散 -> 得到 D（仍在 [0,1)）
        """
        # 1) 归一化：把 uint8 像素 0..255 映射到浮点 0..1
        Pn = img_u8.astype(np.float64) / 255.0
        H, W = Pn.shape

        # 2) 分块：N×8×8
        blocks, (gh, gw) = blockify(Pn, self.B)
        N = blocks.shape[0]

        # 生成混沌序列 z：既用于动态测量矩阵，也用于扩散 mask
        chaos_need = max(N + 4096, gh * gw * self.m_meas + 4096)
        _, _, zs = spcmm_generate_xyz(self.key, chaos_need)

        # 3) 对每块做 DCT，并展平成 64 维 Omega
        Omegas = np.zeros((N, self.n), dtype=np.float64)
        for i in range(N):
            C = dct2(blocks[i])
            Omegas[i] = C.flatten()

        # 4) 动态测量：y_i = Phi_i @ Omega_i
        Y_meas = np.zeros((N, self.m_meas), dtype=np.float64)
        for bi in range(N):
            Phi = phi_from_key_z(self.key, zs[bi], block_id=bi, m_meas=self.m_meas, n=self.n)
            Y_meas[bi] = Phi @ Omegas[bi]

        # 5) 堆叠为 3D 张量：Y(gh,gw,m)
        Y = Y_meas.reshape((gh, gw, self.m_meas))

        # 6) 为了浮点扩散稳定：对 Y 做线性归一化到 [0,1)，并取 frac
        y_min = float(Y.min())
        y_max = float(Y.max())
        denom = (y_max - y_min) if (y_max - y_min) > 1e-12 else 1.0
        Yn = frac_arr((Y - y_min) / denom)

        # 7) mask：从混沌 z 截取一段作为每个点的扰动项
        mask_start = 1024
        flat_mask = zs[mask_start:mask_start + gh * gw * self.m_meas]
        mask = frac_arr(flat_mask.reshape((gh, gw, self.m_meas)))

        # 8) 3D 浮点扩散（核心：让 Yn 的变化全向传播）
        a, b, c = (0.21, 0.33, 0.27)
        D = diffusion3d_forward(Yn, mask, a=a, b=b, c=c)

        # 保存解密必要状态（脚本内复用）
        state = CryptoState(
            H=H, W=W, B=self.B,
            m_meas=self.m_meas,
            grid_h=gh, grid_w=gw,
            y_min=y_min, y_max=y_max,
            diff_abc=(a, b, c),
            mask_start=mask_start,
            chaos_need=chaos_need
        )
        return D.astype(np.float64), state

    def decrypt_from_prequant_D(self, D: np.ndarray, state: CryptoState,
                               fista_lam: float, fista_iter: int):
        """
        解密（从量化前 D 恢复明文图）
        流程：
        1) 复现混沌序列 z 和 mask
        2) 逆浮点扩散：Yn = inv_diffusion(D)
        3) 反归一化：Y = Yn*(y_max-y_min)+y_min
        4) 对每块：用相同 Phi_i 和 FISTA 从 y_i 重建 Omega_i
        5) Omega_i -> 8×8 -> IDCT -> 拼图 -> 得到 P_hat
        """
        gh, gw = state.grid_h, state.grid_w
        assert D.shape == (gh, gw, state.m_meas)

        # 1) 复现混沌序列
        D = frac_arr(D.astype(np.float64))
        _, _, zs = spcmm_generate_xyz(self.key, state.chaos_need)

        # 2) 复现 mask
        flat_mask = zs[state.mask_start:state.mask_start + gh * gw * state.m_meas]
        mask = frac_arr(flat_mask.reshape((gh, gw, state.m_meas)))

        # 3) 逆扩散得到 Yn
        a, b, c = state.diff_abc
        Yn = diffusion3d_inverse(D, mask, a=a, b=b, c=c)

        # 4) 反归一化得到测量张量 Y
        denom = (state.y_max - state.y_min) if (state.y_max - state.y_min) > 1e-12 else 1.0
        Y = Yn * denom + state.y_min

        # 5) 拆成块测量 y_i
        Y_meas = Y.reshape((-1, state.m_meas))
        N = Y_meas.shape[0]
        Omegas_hat = np.zeros((N, self.n), dtype=np.float64)

        # 6) 每块重建（FISTA）
        for bi in range(N):
            Phi = phi_from_key_z(self.key, zs[bi], block_id=bi, m_meas=state.m_meas, n=self.n)
            y = Y_meas[bi]

            # 如果 m=n 且 Phi 是正交的，x≈Phi^T y（快速）
            if state.m_meas == self.n:
                Omegas_hat[bi] = Phi.T @ y
            else:
                Omegas_hat[bi] = fista_l1(Phi, y, lam=fista_lam, max_iter=fista_iter, tol=1e-5)

        # 7) IDCT 得到各块像素，拼回整图
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
# Full measurement module (论文常用指标)
# ============================================================

def mae_u8(a: np.ndarray, b: np.ndarray) -> float:
    """MAE：平均绝对误差"""
    return float(np.mean(np.abs(a.astype(np.int32) - b.astype(np.int32))))

def mse_u8(a: np.ndarray, b: np.ndarray) -> float:
    """MSE"""
    aa = a.astype(np.float64)
    bb = b.astype(np.float64)
    return float(np.mean((aa - bb) ** 2))

def psnr_u8(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR"""
    m = mse_u8(a, b)
    if m < 1e-12:
        return 99.0
    return float(10.0 * np.log10((255.0 ** 2) / m))

def ssim_u8(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM（优先用 skimage）"""
    if ssim_skimage is not None:
        return float(ssim_skimage(a, b, data_range=255))
    # fallback 简单版
    a = a.astype(np.float64); b = b.astype(np.float64)
    mu_x = a.mean(); mu_y = b.mean()
    var_x = a.var(); var_y = b.var()
    cov_xy = ((a - mu_x) * (b - mu_y)).mean()
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    return float(((2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)) /
                 ((mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2)))

def entropy_u8(img: np.ndarray) -> float:
    """信息熵（越接近 8 越均匀）"""
    hist = np.bincount(img.flatten(), minlength=256).astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def npcr_u8(c1: np.ndarray, c2: np.ndarray) -> float:
    """NPCR：像素变化率（越接近 100% 越好）"""
    return float(np.mean(c1 != c2) * 100.0)

def uaci_u8(c1: np.ndarray, c2: np.ndarray) -> float:
    """UACI：平均变化强度（常见理想值 ~33%）"""
    return float(np.mean(np.abs(c1.astype(np.int32) - c2.astype(np.int32))) / 255.0 * 100.0)

def corr_adjacent(img: np.ndarray, mode: str = "h", samples: int = 200000, seed: int = 0) -> float:
    """
    相邻像素相关性（h/v/d）
    越接近 0 表示密文无相邻相关性
    """
    rng = np.random.default_rng(seed)
    H, W = img.shape
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
    """保存直方图曲线"""
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
    """保存相关散点图（论文常用）"""
    rng = np.random.default_rng(seed)
    H, W = img_u8.shape
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
    args = ap.parse_args()

    # 固定块大小 B=8、每块 DCT 维度 n=64
    B = 8
    n = 64

    # 压缩率 cr = m/64, 计算 m_meas
    cr = float(args.cr)
    cr = max(1.0 / n, min(1.0, cr))
    m_meas = int(np.clip(int(round(cr * n)), 1, n))

    ensure_dir(args.out)

    # 读入灰度图
    img = Image.open(args.img).convert("L")
    img = np.array(img, dtype=np.uint8)

    # 保证 H,W 是 8 的整数倍（否则裁剪）
    H, W = img.shape
    H2, W2 = (H // B) * B, (W // B) * B
    img = img[:H2, :W2]
    H, W = img.shape

    KEY = args.key
    WRONG_KEY = args.wrong_key
    KEY_FLIP = flip_one_bit_in_key(KEY)

    print("=== Config ===")
    print(f"image: {args.img}  size: {H}x{W}")
    print(f"block: {B}x{B}  DCT n=64")
    print(f"compression rate cr=m/64: {cr:.4f}  -> m_meas={m_meas}")
    print(f"solver: fista   lam: {args.lam}   iter: {args.iter}")
    print(f"Numba enabled: {NUMBA_OK}")
    print("Dynamic S-box per depth: True\n")

    crypto = CSChaosCrypto(KEY, B=B, m_meas=m_meas)

    # ========================================================
    # Encrypt
    # ========================================================
    t0 = time.perf_counter()

    # 1) 先做“量化前浮点域加密”，得到 D（真·测量域密文浮点张量）
    D, state = crypto.encrypt_prequant_D(img)

    # 2) D -> uint32（离散化，进入可逆密文域）
    U = u32_from_float01(D)

    # 3) uint32 域 ARX + S-box 扩散 -> 得到真实密文 Cu32
    Cu32 = u32_diffuse_forward(U, KEY)

    t1 = time.perf_counter()

    enc_time = t1 - t0
    plain_bytes = img.size
    enc_thr = (plain_bytes / (1024 * 1024)) / max(enc_time, 1e-12)

    # 4) 把 Cu32 封装为可逆的 RGBA png（最终“密文图”）
    cipher_rgba_path = os.path.join(args.out, "cipher_rgba.png")
    save_cipher_rgba(Cu32, cipher_rgba_path)

    # 5) 生成一个不可逆的 uint8 view 用于统计、可视化
    cipher_u8 = cipher_uint8_view_from_Cu32(Cu32, H, W)
    cipher_u8_path = os.path.join(args.out, "cipher_uint8.png")
    Image.fromarray(cipher_u8).save(cipher_u8_path)

    # ========================================================
    # Decrypt from cipher_rgba.png  (correct key)
    # ========================================================
    # 从 RGBA 密文图还原 Cu32（应完全一致）
    gh, gw, mm = state.grid_h, state.grid_w, state.m_meas
    Cu32_from_img = load_cipher_rgba_to_Cu32(cipher_rgba_path, gh, gw, mm)
    cu32_img_mismatch = int(np.count_nonzero(Cu32_from_img != Cu32))

    t2 = time.perf_counter()

    # 1) 逆 uint32 ARX + S-box 扩散：Cu32 -> U
    U_test = u32_diffuse_inverse(Cu32_from_img, KEY)

    # 2) 反量化：U -> D_back（float [0,1)）
    D_back = float01_from_u32(U_test)

    # 3) 逆浮点扩散 + CS 重建 + IDCT -> rec
    rec = crypto.decrypt_from_prequant_D(D_back, state, fista_lam=args.lam, fista_iter=args.iter)

    t3 = time.perf_counter()

    dec_time = t3 - t2
    dec_thr = (plain_bytes / (1024 * 1024)) / max(dec_time, 1e-12)

    # 检查 U 是否可逆回到原 U（应为 0）
    mismatch = int(np.count_nonzero(U_test != U))

    # ========================================================
    # Save base outputs
    # ========================================================
    Image.fromarray(img).save(os.path.join(args.out, "plain.png"))
    Image.fromarray(rec).save(os.path.join(args.out, "decrypted.png"))
    diff = np.abs(rec.astype(np.int32) - img.astype(np.int32))
    diff_vis = np.clip(diff, 0, 255).astype(np.uint8)
    Image.fromarray(diff_vis).save(os.path.join(args.out, "abs_diff.png"))

    # 保存真实密文与中间浮点 D（便于复现实验）
    np.save(os.path.join(args.out, "cipher_u32.npy"), Cu32)
    np.save(os.path.join(args.out, "cipher_D.npy"), D)

    # ========================================================
    # Reconstruction quality
    # ========================================================
    MAE = mae_u8(rec, img)
    PSNR = psnr_u8(rec, img)
    SSIM = ssim_u8(rec, img)

    # ========================================================
    # Cipher stats (cipher_uint8 view)
    # ========================================================
    ENT = entropy_u8(cipher_u8)
    CH = corr_adjacent(cipher_u8, "h", seed=1)
    CV = corr_adjacent(cipher_u8, "v", seed=2)
    CD = corr_adjacent(cipher_u8, "d", seed=3)

    # 绘图输出：直方图 & 相关散点
    save_histogram(img, os.path.join(args.out, "hist_plain.png"), "Histogram (Plain)")
    save_histogram(cipher_u8, os.path.join(args.out, "hist_cipher.png"), "Histogram (Cipher uint8 view)")
    save_histogram(rec, os.path.join(args.out, "hist_decrypted.png"), "Histogram (Decrypted)")

    save_corr_scatter(cipher_u8, os.path.join(args.out, "corr_cipher_h.png"), "h", seed=10)
    save_corr_scatter(cipher_u8, os.path.join(args.out, "corr_cipher_v.png"), "v", seed=11)
    save_corr_scatter(cipher_u8, os.path.join(args.out, "corr_cipher_d.png"), "d", seed=12)

    # ========================================================
    # Differential test (flip 1 plaintext pixel)
    # ========================================================
    # 只改动明文一个像素，再加密，比较密文变化（NPCR/UACI）
    img2 = img.copy()
    img2[0, 0] = np.uint8((int(img2[0, 0]) + 1) % 256)

    D2, _ = crypto.encrypt_prequant_D(img2)
    U2 = u32_from_float01(D2)
    Cu32_2 = u32_diffuse_forward(U2, KEY)
    cipher_u8_2 = cipher_uint8_view_from_Cu32(Cu32_2, H, W)

    NPCR = npcr_u8(cipher_u8, cipher_u8_2)
    UACI = uaci_u8(cipher_u8, cipher_u8_2)

    cipher_diff = np.abs(cipher_u8.astype(np.int16) - cipher_u8_2.astype(np.int16))
    Image.fromarray(np.clip(cipher_diff, 0, 255).astype(np.uint8)).save(os.path.join(args.out, "cipher_diff_abs.png"))

    # ========================================================
    # Key sensitivity test (flip 1 key bit)
    # ========================================================
    # 明文不变，只改变 key 的 1 bit，看密文变化（NPCR/UACI）
    Dk, _ = crypto.encrypt_prequant_D(img)
    Uk = u32_from_float01(Dk)
    Cu32_k = u32_diffuse_forward(Uk, KEY_FLIP)
    cipher_u8_k = cipher_uint8_view_from_Cu32(Cu32_k, H, W)

    NPCR_key = npcr_u8(cipher_u8, cipher_u8_k)
    UACI_key = uaci_u8(cipher_u8, cipher_u8_k)
    Image.fromarray(cipher_u8_k).save(os.path.join(args.out, "cipher_uint8_keyflip.png"))

    # ========================================================
    # Wrong-key decryption test
    # ========================================================
    # 用错误 key 逆 ARX，然后尝试重建，应得到“接近随机”的结果
    U_wrong = u32_diffuse_inverse(Cu32_from_img, WRONG_KEY)
    D_wrong = float01_from_u32(U_wrong)

    crypto_wrong = CSChaosCrypto(WRONG_KEY, B=B, m_meas=m_meas)
    rec_wrong = crypto_wrong.decrypt_from_prequant_D(D_wrong, state, fista_lam=args.lam, fista_iter=args.iter)
    Image.fromarray(rec_wrong).save(os.path.join(args.out, "decrypted_wrong_key.png"))

    MAE_wrong = mae_u8(rec_wrong, img)
    PSNR_wrong = psnr_u8(rec_wrong, img)
    SSIM_wrong = ssim_u8(rec_wrong, img)

    # ========================================================
    # Report
    # ========================================================
    lines = []
    lines.append("=== Config ===")
    lines.append(f"image: {args.img}  size: {H}x{W}")
    lines.append(f"block: {B}x{B}  DCT n=64")
    lines.append(f"compression rate cr=m/64: {cr:.4f}  -> m_meas={m_meas}")
    lines.append(f"solver: fista   lam: {args.lam}   iter: {args.iter}")
    lines.append(f"Numba enabled: {NUMBA_OK}")
    lines.append("Dynamic S-box per depth: True\n")

    lines.append("=== Encrypt ===")
    lines.append(f"D shape: {D.shape} float64 range: {float(D.min())} {float(D.max())}")
    lines.append(f"Cu32 shape: {Cu32.shape} uint32 (REAL ciphertext)")
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

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.out, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

if __name__ == "__main__":
    main()
