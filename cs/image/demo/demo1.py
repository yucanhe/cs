"""
CS + 3D float diffusion + uint32 ARX diffusion WITH S-BOX (both directions)
AND reversible final cipher image (RGBA PNG) for real decryption from the final cipher image.

Key points:
- cipher_uint8.png (byte-mix) is ONLY for visualization/metrics (not invertible)
- cipher_rgba.png stores REAL ciphertext words (uint32) as RGBA bytes -> invertible
- Decrypt reads cipher_rgba.png -> reconstruct Cu32 -> full decryption pipeline
- Compression rate adjustable: --cr, with OMP reconstruction when m<64
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

try:
    from skimage.metrics import structural_similarity as ssim_skimage
except Exception:
    ssim_skimage = None


# ============================================================
# Utils
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def frac(x: np.ndarray) -> np.ndarray:
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


# ============================================================
# Chaos generator (demo; replace with your true 3D-SPCMM if needed)
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
        x_next = frac(a * x * (1 - x) + k1 * np.sin(np.pi * (y + z)))
        y_next = frac(b * y * (1 - y) + k2 * np.sin(np.pi * (x + z)))
        z_next = frac(c * z * (1 - z) + k3 * np.sin(np.pi * (x + y)))
        x, y, z = x_next, y_next, z_next
        if t >= burn:
            i = t - burn
            xs[i], ys[i], zs[i] = x, y, z
    return xs, ys, zs


# ============================================================
# Dynamic Phi
# ============================================================

def phi_from_key_z(key: str, z_val: float, block_id: int, m_meas: int, n: int = 64):
    """
    Phi_i in R^{m_meas x n}
    If m_meas == n: make it orthogonal (QR).
    Else: normalized random Gaussian rows.
    """
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


# ============================================================
# Float 3D diffusion (invertible via frac)
# ============================================================

def diffusion3d_forward(Y: np.ndarray, mask: np.ndarray, a=0.21, b=0.33, c=0.27):
    H, W, M = Y.shape
    D = np.empty_like(Y, dtype=np.float64)
    for i in range(H):
        for j in range(W):
            for k in range(M):
                up   = D[i-1, j, k] if i > 0 else 0.0
                left = D[i, j-1, k] if j > 0 else 0.0
                prev = D[i, j, k-1] if k > 0 else 0.0
                D[i, j, k] = frac(Y[i, j, k] + a*up + b*left + c*prev + mask[i, j, k])
    return D

def diffusion3d_inverse(D: np.ndarray, mask: np.ndarray, a=0.21, b=0.33, c=0.27):
    H, W, M = D.shape
    Y = np.empty_like(D, dtype=np.float64)
    for i in range(H):
        for j in range(W):
            for k in range(M):
                up   = D[i-1, j, k] if i > 0 else 0.0
                left = D[i, j-1, k] if j > 0 else 0.0
                prev = D[i, j, k-1] if k > 0 else 0.0
                Y[i, j, k] = frac(D[i, j, k] - a*up - b*left - c*prev - mask[i, j, k])
    return Y


# ============================================================
# uint32 ARX diffusion + S-box (REAL ciphertext)
# ============================================================

UINT32_MOD = 2**32
MOD_MASK = 0xFFFFFFFF

def u32_from_float01(D: np.ndarray) -> np.ndarray:
    D = frac(D.astype(np.float64))
    return np.floor(D * UINT32_MOD).astype(np.uint32)

def float01_from_u32(U: np.ndarray) -> np.ndarray:
    return (U.astype(np.float64) / UINT32_MOD)

def u32_keystream(key: str, shape) -> np.ndarray:
    seed = _hash_to_seed(key)
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**32, size=np.prod(shape), dtype=np.uint32).reshape(shape)

def rotl32(x: int, r: int) -> int:
    r &= 31
    return ((x << r) | (x >> (32 - r))) & MOD_MASK

def rotr32(x: int, r: int) -> int:
    r &= 31
    return ((x >> r) | (x << (32 - r))) & MOD_MASK

def build_sbox_from_chaos(key: str, label: str = "sbox"):
    """
    Build a 256-permutation S-box using chaotic-like sequence:
    generate 256 values from spcmm and sort -> permutation.
    """
    xs, ys, zs = spcmm_generate_xyz(key + "|" + label, 256 + 64)
    seq = xs[:256]
    perm = np.argsort(seq).astype(np.uint8)  # values 0..255 in some order
    sbox = perm.copy()
    inv = np.empty(256, dtype=np.uint8)
    inv[sbox] = np.arange(256, dtype=np.uint8)
    return sbox, inv

def sub_bytes_u32(x: int, sbox: np.ndarray) -> int:
    b0 = (x >> 24) & 0xFF
    b1 = (x >> 16) & 0xFF
    b2 = (x >>  8) & 0xFF
    b3 = (x      ) & 0xFF
    y0 = int(sbox[b0])
    y1 = int(sbox[b1])
    y2 = int(sbox[b2])
    y3 = int(sbox[b3])
    return ((y0 << 24) | (y1 << 16) | (y2 << 8) | y3) & MOD_MASK

def u32_diffuse_forward(U: np.ndarray, key: str, use_sbox: bool = True) -> np.ndarray:
    """
    Pass1 (forward): ARX + optional SubBytes
    Pass2 (backward): ARX + optional SubBytes
    """
    H, W, M = U.shape

    if use_sbox:
        sbox1, _ = build_sbox_from_chaos(key, "sbox_pass1")
        sbox2, _ = build_sbox_from_chaos(key, "sbox_pass2")
    else:
        sbox1 = sbox2 = None

    mask1 = u32_keystream(key + "|mask1|", U.shape)
    ks1   = u32_keystream(key + "|ks1|",   U.shape)
    C = np.empty_like(U, dtype=np.uint32)

    # Pass1 forward: uses -1 neighbors
    for i in range(H):
        for j in range(W):
            for k in range(M):
                up   = int(C[i-1, j, k]) if i > 0 else 0
                left = int(C[i, j-1, k]) if j > 0 else 0
                prev = int(C[i, j, k-1]) if k > 0 else 0

                s = (int(U[i, j, k])
                     + rotl32(up, 7) + rotl32(left, 11) + rotl32(prev, 19)
                     + int(mask1[i, j, k])) & MOD_MASK

                t = rotl32(s, 3) ^ int(ks1[i, j, k])
                if sbox1 is not None:
                    t = sub_bytes_u32(t, sbox1)
                C[i, j, k] = np.uint32(t)

    # Pass2 backward: uses +1 neighbors
    mask2 = u32_keystream(key + "|mask2|", U.shape)
    ks2   = u32_keystream(key + "|ks2|",   U.shape)

    for i in range(H - 1, -1, -1):
        for j in range(W - 1, -1, -1):
            for k in range(M - 1, -1, -1):
                dn  = int(C[i+1, j, k]) if i + 1 < H else 0
                rt  = int(C[i, j+1, k]) if j + 1 < W else 0
                nxt = int(C[i, j, k+1]) if k + 1 < M else 0

                s = (int(C[i, j, k])
                     + rotl32(dn, 5) + rotl32(rt, 13) + rotl32(nxt, 17)
                     + int(mask2[i, j, k])) & MOD_MASK

                t = rotl32(s, 9) ^ int(ks2[i, j, k])
                if sbox2 is not None:
                    t = sub_bytes_u32(t, sbox2)
                C[i, j, k] = np.uint32(t)

    return C

def u32_diffuse_inverse(C: np.ndarray, key: str, use_sbox: bool = True) -> np.ndarray:
    """
    Correct inverse ordering:
      Undo Pass2 in FORWARD scan
      Undo Pass1 in BACKWARD scan

    With S-box:
      In forward: SubBytes applied at the END of each pass.
      So inverse must apply InvSubBytes FIRST.
    """
    H, W, M = C.shape

    if use_sbox:
        _, inv1 = build_sbox_from_chaos(key, "sbox_pass1")
        _, inv2 = build_sbox_from_chaos(key, "sbox_pass2")
    else:
        inv1 = inv2 = None

    # ---- Undo Pass2 (FORWARD scan!)
    mask2 = u32_keystream(key + "|mask2|", C.shape)
    ks2   = u32_keystream(key + "|ks2|",   C.shape)

    C1 = C.copy().astype(np.uint32)

    for i in range(H):
        for j in range(W):
            for k in range(M):
                dn  = int(C1[i+1, j, k]) if i + 1 < H else 0
                rt  = int(C1[i, j+1, k]) if j + 1 < W else 0
                nxt = int(C1[i, j, k+1]) if k + 1 < M else 0

                t = int(C1[i, j, k])
                if inv2 is not None:
                    t = sub_bytes_u32(t, inv2)
                t ^= int(ks2[i, j, k])
                s = rotr32(t, 9)

                orig = (s
                        - rotl32(dn, 5) - rotl32(rt, 13) - rotl32(nxt, 17)
                        - int(mask2[i, j, k])) & MOD_MASK

                C1[i, j, k] = np.uint32(orig)

    # ---- Undo Pass1 (BACKWARD scan!)
    mask1 = u32_keystream(key + "|mask1|", C.shape)
    ks1   = u32_keystream(key + "|ks1|",   C.shape)

    U = np.empty_like(C1, dtype=np.uint32)

    for i in range(H - 1, -1, -1):
        for j in range(W - 1, -1, -1):
            for k in range(M - 1, -1, -1):
                up   = int(C1[i-1, j, k]) if i > 0 else 0
                left = int(C1[i, j-1, k]) if j > 0 else 0
                prev = int(C1[i, j, k-1]) if k > 0 else 0

                t = int(C1[i, j, k])
                if inv1 is not None:
                    t = sub_bytes_u32(t, inv1)
                t ^= int(ks1[i, j, k])
                s = rotr32(t, 3)

                u = (s
                     - rotl32(up, 7) - rotl32(left, 11) - rotl32(prev, 19)
                     - int(mask1[i, j, k])) & MOD_MASK

                U[i, j, k] = np.uint32(u)

    return U


# ============================================================
# Cipher images
# ============================================================

def cipher_uint8_view_from_Cu32(Cu32: np.ndarray, out_H: int, out_W: int):
    """
    NON-invertible view (byte-mix) for stats only.
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
    Invertible final cipher image:
    Each uint32 word -> 4 bytes -> RGBA pixel.
    Image shape: (H', W'*m) with mode RGBA.
    """
    H, W, M = Cu32.shape
    words = Cu32.reshape(-1).astype(np.uint32)
    b0 = ((words >> 24) & 0xFF).astype(np.uint8)
    b1 = ((words >> 16) & 0xFF).astype(np.uint8)
    b2 = ((words >> 8) & 0xFF).astype(np.uint8)
    b3 = (words & 0xFF).astype(np.uint8)
    rgba = np.stack([b0, b1, b2, b3], axis=1)  # [N,4]

    # reshape to (H, W*M, 4)
    img_rgba = rgba.reshape(H, W * M, 4)
    Image.fromarray(img_rgba, mode="RGBA").save(out_path)

def load_cipher_rgba_to_Cu32(path: str, H_: int, W_: int, M_: int) -> np.ndarray:
    """
    Read invertible cipher image back into Cu32 of shape (H', W', M).
    Expects image shape (H', W'*M) RGBA.
    """
    im = Image.open(path).convert("RGBA")
    arr = np.array(im, dtype=np.uint8)
    if arr.shape[0] != H_ or arr.shape[1] != W_ * M_ or arr.shape[2] != 4:
        raise ValueError(f"cipher_rgba shape mismatch: got {arr.shape}, expect ({H_},{W_*M_},4)")
    flat = arr.reshape(-1, 4).astype(np.uint32)
    words = (flat[:,0] << 24) | (flat[:,1] << 16) | (flat[:,2] << 8) | flat[:,3]
    return words.reshape(H_, W_, M_).astype(np.uint32)


# ============================================================
# OMP reconstruction (for m < 64)
# ============================================================

def omp(Phi: np.ndarray, y: np.ndarray, k: int, tol: float = 1e-6) -> np.ndarray:
    """
    Solve y = Phi x, assuming x is k-sparse (canonical basis).
    Basic OMP:
      - pick atom with max correlation
      - LS on selected atoms
      - stop when k atoms selected or residual small
    """
    m, n = Phi.shape
    y = y.astype(np.float64)
    r = y.copy()
    support = []
    x = np.zeros(n, dtype=np.float64)

    PhiT = Phi.T
    sol = None
    for _ in range(min(k, n)):
        corr = np.abs(PhiT @ r)
        j = int(np.argmax(corr))
        if j in support:
            break
        support.append(j)

        A = Phi[:, support]
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ sol
        if np.linalg.norm(r) < tol:
            break

    if support and sol is not None:
        x[support] = sol
    return x


# ============================================================
# Crypto state + core (no side info beyond public params + key)
# ============================================================

@dataclass
class CryptoState:
    H: int
    W: int
    B: int
    m_meas: int
    grid_h: int
    grid_w: int
    perm: np.ndarray
    inv_perm: np.ndarray
    y_min: float
    y_max: float
    diff_abc: tuple
    mask_start: int
    chaos_need: int

class CSChaosCryptoAdjustableCR:
    def __init__(self, key: str, B=8, m_meas=64):
        self.key = key
        self.B = B
        self.n = 64  # DCT coeff length
        self.m_meas = int(m_meas)

    def encrypt_prequant_D(self, img_u8: np.ndarray):
        Pn = img_u8.astype(np.float64) / 255.0
        H, W = Pn.shape
        blocks, (gh, gw) = blockify(Pn, self.B)
        N = blocks.shape[0]

        chaos_need = max(N + 4096, gh * gw * self.m_meas + 4096)
        xs, ys, zs = spcmm_generate_xyz(self.key, chaos_need)

        # full 64 DCT coeff vector per block
        Omegas = np.zeros((N, self.n), dtype=np.float64)
        for i in range(N):
            C = dct2(blocks[i])
            Omegas[i] = C.flatten()

        # inter-block permutation
        perm = np.argsort(xs[:N])
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(N)
        Omegas_p = Omegas[perm]

        # dynamic measurement
        Y_meas = np.zeros((N, self.m_meas), dtype=np.float64)
        for bi in range(N):
            Phi = phi_from_key_z(self.key, zs[bi], block_id=bi, m_meas=self.m_meas, n=self.n)
            Y_meas[bi] = Phi @ Omegas_p[bi]

        Y_tensor = Y_meas.reshape((gh, gw, self.m_meas))

        # normalize into (0,1)
        y_min = float(Y_tensor.min())
        y_max = float(Y_tensor.max())
        denom = (y_max - y_min) if (y_max - y_min) > 1e-12 else 1.0
        Yn = frac((Y_tensor - y_min) / denom)

        # mask
        mask_start = 1024
        flat_mask = zs[mask_start:mask_start + gh * gw * self.m_meas]
        mask = frac(flat_mask.reshape((gh, gw, self.m_meas)))

        a, b, c = (0.21, 0.33, 0.27)
        D = diffusion3d_forward(Yn, mask, a=a, b=b, c=c)

        state = CryptoState(
            H=H, W=W, B=self.B,
            m_meas=self.m_meas,
            grid_h=gh, grid_w=gw,
            perm=perm, inv_perm=inv_perm,
            y_min=y_min, y_max=y_max,
            diff_abc=(a, b, c),
            mask_start=mask_start,
            chaos_need=chaos_need
        )
        return D.astype(np.float64), state

    def decrypt_from_prequant_D(self, D: np.ndarray, state: CryptoState, omp_k: int = 16):
        gh, gw = state.grid_h, state.grid_w
        assert D.shape == (gh, gw, state.m_meas)

        D = frac(D.astype(np.float64))
        xs, ys, zs = spcmm_generate_xyz(self.key, state.chaos_need)

        # reconstruct mask
        flat_mask = zs[state.mask_start:state.mask_start + gh * gw * state.m_meas]
        mask = frac(flat_mask.reshape((gh, gw, state.m_meas)))

        a, b, c = state.diff_abc
        Yn = diffusion3d_inverse(D, mask, a=a, b=b, c=c)

        denom = (state.y_max - state.y_min) if (state.y_max - state.y_min) > 1e-12 else 1.0
        Y_tensor = Yn * denom + state.y_min

        Y_meas = Y_tensor.reshape((-1, state.m_meas))
        N = Y_meas.shape[0]

        Omegas_p_hat = np.zeros((N, self.n), dtype=np.float64)
        for bi in range(N):
            Phi = phi_from_key_z(self.key, zs[bi], block_id=bi, m_meas=state.m_meas, n=self.n)
            y = Y_meas[bi]

            if state.m_meas == self.n:
                Omegas_p_hat[bi] = Phi.T @ y
            else:
                Omegas_p_hat[bi] = omp(Phi, y, k=omp_k)

        # inverse perm
        Omegas_hat = Omegas_p_hat[state.inv_perm]

        # IDCT per block
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
# Metrics + plots
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
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_x = a.mean()
    mu_y = b.mean()
    var_x = a.var()
    var_y = b.var()
    cov_xy = ((a - mu_x) * (b - mu_y)).mean()
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    return float(((2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2)))

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
    if mode == "h":
        xs = rng.integers(0, H, size=samples)
        ys = rng.integers(0, W - 1, size=samples)
        p1 = img[xs, ys].astype(np.float64)
        p2 = img[xs, ys + 1].astype(np.float64)
    elif mode == "v":
        xs = rng.integers(0, H - 1, size=samples)
        ys = rng.integers(0, W, size=samples)
        p1 = img[xs, ys].astype(np.float64)
        p2 = img[xs + 1, ys].astype(np.float64)
    elif mode == "d":
        xs = rng.integers(0, H - 1, size=samples)
        ys = rng.integers(0, W - 1, size=samples)
        p1 = img[xs, ys].astype(np.float64)
        p2 = img[xs + 1, ys + 1].astype(np.float64)
    else:
        raise ValueError("mode must be 'h','v','d'")
    p1m = p1 - p1.mean()
    p2m = p2 - p2.mean()
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

    if mode == "h":
        xs = rng.integers(0, H, size=samples)
        ys = rng.integers(0, W - 1, size=samples)
        p1 = img_u8[xs, ys]
        p2 = img_u8[xs, ys + 1]
        title = "Adjacent Correlation Scatter (Horizontal)"
    elif mode == "v":
        xs = rng.integers(0, H - 1, size=samples)
        ys = rng.integers(0, W, size=samples)
        p1 = img_u8[xs, ys]
        p2 = img_u8[xs + 1, ys]
        title = "Adjacent Correlation Scatter (Vertical)"
    elif mode == "d":
        xs = rng.integers(0, H - 1, size=samples)
        ys = rng.integers(0, W - 1, size=samples)
        p1 = img_u8[xs, ys]
        p2 = img_u8[xs + 1, ys + 1]
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
    ap.add_argument("--out", type=str, default="outputs_sbox_rgba")
    ap.add_argument("--key", type=str, default="my-secret-key-2026")
    ap.add_argument("--wrong_key", type=str, default="my-secret-key-2026_wrong")
    ap.add_argument("--cr", type=float, default=1.0, help="compression rate m/64 in (0,1]")
    ap.add_argument("--omp_k", type=int, default=16, help="OMP sparsity (used when cr<1)")
    ap.add_argument("--no_sbox", action="store_true", help="disable s-box (for ablation)")
    args = ap.parse_args()

    B = 8
    n = 64
    cr = float(args.cr)
    cr = max(1.0 / n, min(1.0, cr))
    m_meas = int(np.clip(int(round(cr * n)), 1, n))

    ensure_dir(args.out)

    # load image
    img = Image.open(args.img).convert("L")
    img = np.array(img, dtype=np.uint8)

    # crop to multiple of 8
    H, W = img.shape
    H2, W2 = (H // B) * B, (W // B) * B
    img = img[:H2, :W2]
    H, W = img.shape

    KEY = args.key
    WRONG_KEY = args.wrong_key
    KEY_FLIP = flip_one_bit_in_key(KEY)
    use_sbox = (not args.no_sbox)

    crypto = CSChaosCryptoAdjustableCR(KEY, B=B, m_meas=m_meas)

    # =========================
    # Encrypt
    # =========================
    t0 = time.perf_counter()
    D, state = crypto.encrypt_prequant_D(img)
    U = u32_from_float01(D)
    Cu32 = u32_diffuse_forward(U, KEY, use_sbox=use_sbox)
    t1 = time.perf_counter()

    enc_time = t1 - t0
    plain_bytes = img.size
    enc_thr = (plain_bytes / (1024 * 1024)) / max(enc_time, 1e-12)

    # Save final reversible cipher image (RGBA)
    cipher_rgba_path = os.path.join(args.out, "cipher_rgba.png")
    save_cipher_rgba(Cu32, cipher_rgba_path)

    # Non-invertible view for stats
    cipher_u8 = cipher_uint8_view_from_Cu32(Cu32, H, W)

    # =========================
    # Decrypt FROM FINAL CIPHER IMAGE (RGBA)
    # =========================
    gh, gw, mm = state.grid_h, state.grid_w, state.m_meas
    Cu32_from_img = load_cipher_rgba_to_Cu32(cipher_rgba_path, gh, gw, mm)

    # sanity check integer invertibility
    U_test = u32_diffuse_inverse(Cu32_from_img, KEY, use_sbox=use_sbox)
    mismatch = int(np.count_nonzero(U_test != U))

    t2 = time.perf_counter()
    D_back = float01_from_u32(U_test)
    rec = crypto.decrypt_from_prequant_D(D_back, state, omp_k=args.omp_k)
    t3 = time.perf_counter()

    dec_time = t3 - t2
    dec_thr = (plain_bytes / (1024 * 1024)) / max(dec_time, 1e-12)

    # save base outputs
    Image.fromarray(img).save(os.path.join(args.out, "plain.png"))
    Image.fromarray(cipher_u8).save(os.path.join(args.out, "cipher_uint8.png"))  # stats view
    Image.fromarray(rec).save(os.path.join(args.out, "decrypted.png"))
    diff = np.abs(rec.astype(np.int32) - img.astype(np.int32))
    diff_vis = np.clip(diff, 0, 255).astype(np.uint8)
    Image.fromarray(diff_vis).save(os.path.join(args.out, "abs_diff.png"))
    np.save(os.path.join(args.out, "cipher_u32.npy"), Cu32)
    np.save(os.path.join(args.out, "cipher_D.npy"), D)

    # reconstruction metrics
    MAE = mae_u8(rec, img)
    PSNR = psnr_u8(rec, img)
    SSIM = ssim_u8(rec, img)

    # cipher stats (on cipher_uint8 view)
    ENT = entropy_u8(cipher_u8)
    CH = corr_adjacent(cipher_u8, "h", seed=1)
    CV = corr_adjacent(cipher_u8, "v", seed=2)
    CD = corr_adjacent(cipher_u8, "d", seed=3)

    # plots
    save_histogram(img, os.path.join(args.out, "hist_plain.png"), "Histogram (Plain)")
    save_histogram(cipher_u8, os.path.join(args.out, "hist_cipher.png"), "Histogram (Cipher uint8 view)")
    save_histogram(rec, os.path.join(args.out, "hist_decrypted.png"), "Histogram (Decrypted)")

    save_corr_scatter(cipher_u8, os.path.join(args.out, "corr_cipher_h.png"), "h", seed=10)
    save_corr_scatter(cipher_u8, os.path.join(args.out, "corr_cipher_v.png"), "v", seed=11)
    save_corr_scatter(cipher_u8, os.path.join(args.out, "corr_cipher_d.png"), "d", seed=12)

    # =========================
    # Differential test (flip 1 plaintext pixel) ON cipher_uint8 view
    # =========================
    img2 = img.copy()
    img2[0, 0] = np.uint8((int(img2[0, 0]) + 1) % 256)

    D2, _ = crypto.encrypt_prequant_D(img2)
    Cu32_2 = u32_diffuse_forward(u32_from_float01(D2), KEY, use_sbox=use_sbox)
    cipher_u8_2 = cipher_uint8_view_from_Cu32(Cu32_2, H, W)

    NPCR = npcr_u8(cipher_u8, cipher_u8_2)
    UACI = uaci_u8(cipher_u8, cipher_u8_2)

    cipher_diff = np.abs(cipher_u8.astype(np.int16) - cipher_u8_2.astype(np.int16))
    cipher_diff_vis = np.clip(cipher_diff, 0, 255).astype(np.uint8)
    Image.fromarray(cipher_diff_vis).save(os.path.join(args.out, "cipher_diff_abs.png"))

    # =========================
    # Key sensitivity test (flip 1 bit in key)
    # =========================
    Dk, _ = crypto.encrypt_prequant_D(img)
    Cu32_k = u32_diffuse_forward(u32_from_float01(Dk), KEY_FLIP, use_sbox=use_sbox)
    cipher_u8_k = cipher_uint8_view_from_Cu32(Cu32_k, H, W)

    NPCR_key = npcr_u8(cipher_u8, cipher_u8_k)
    UACI_key = uaci_u8(cipher_u8, cipher_u8_k)
    Image.fromarray(cipher_u8_k).save(os.path.join(args.out, "cipher_uint8_keyflip.png"))

    # =========================
    # Wrong-key decryption FROM cipher_rgba.png
    # =========================
    U_wrong = u32_diffuse_inverse(Cu32_from_img, WRONG_KEY, use_sbox=use_sbox)
    D_wrong = float01_from_u32(U_wrong)
    crypto_wrong = CSChaosCryptoAdjustableCR(WRONG_KEY, B=B, m_meas=m_meas)
    rec_wrong = crypto_wrong.decrypt_from_prequant_D(D_wrong, state, omp_k=args.omp_k)
    Image.fromarray(rec_wrong).save(os.path.join(args.out, "decrypted_wrong_key.png"))

    MAE_wrong = mae_u8(rec_wrong, img)
    PSNR_wrong = psnr_u8(rec_wrong, img)
    SSIM_wrong = ssim_u8(rec_wrong, img)

    # =========================
    # Report
    # =========================
    lines = []
    lines.append("=== Config ===")
    lines.append(f"image: {args.img}  size: {H}x{W}")
    lines.append(f"block: {B}x{B}  DCT n=64")
    lines.append(f"compression rate cr=m/64: {cr:.4f}  -> m_meas={m_meas}")
    lines.append(f"OMP k (used if m<64): {args.omp_k}")
    lines.append(f"S-box enabled: {use_sbox}")
    lines.append("")
    lines.append("=== Encrypt ===")
    lines.append(f"D shape: {D.shape} float64 range: {float(D.min())} {float(D.max())}")
    lines.append(f"Cu32 shape: {Cu32.shape} uint32 (REAL ciphertext)")
    lines.append(f"Encrypt time: {enc_time:.6f}s  throughput: {enc_thr:.3f} MiB/s")
    lines.append(f"cipher_rgba.png saved: {cipher_rgba_path} (INVERTIBLE FINAL CIPHER IMAGE)")
    lines.append("")
    lines.append("=== Decrypt FROM cipher_rgba.png (correct key) ===")
    lines.append(f"u32 invertibility mismatch count: {mismatch} (should be 0)")
    lines.append(f"Decrypt time: {dec_time:.6f}s  throughput: {dec_thr:.3f} MiB/s")
    lines.append("")
    lines.append("=== Reconstruction quality (plain vs decrypted) ===")
    lines.append(f"MAE:  {MAE}")
    lines.append(f"PSNR: {PSNR}")
    lines.append(f"SSIM: {SSIM}")
    lines.append("")
    lines.append("=== Cipher statistics ON cipher_uint8.png (view derived from Cu32) ===")
    lines.append(f"cipher_uint8 shape: {cipher_u8.shape}")
    lines.append(f"Entropy: {ENT:.6f} bits")
    lines.append(f"AdjCorr-H: {CH:.6f}")
    lines.append(f"AdjCorr-V: {CV:.6f}")
    lines.append(f"AdjCorr-D: {CD:.6f}")
    lines.append("")
    lines.append("=== Differential test (flip 1 plaintext pixel) ON cipher_uint8 view ===")
    lines.append(f"NPCR: {NPCR:.6f}%")
    lines.append(f"UACI: {UACI:.6f}%")
    lines.append("")
    lines.append("=== Key sensitivity test (flip 1 key bit) ON cipher_uint8 view ===")
    lines.append(f"Key original: {KEY}")
    lines.append(f"Key flipped:  {KEY_FLIP}")
    lines.append(f"NPCR_key: {NPCR_key:.6f}%")
    lines.append(f"UACI_key: {UACI_key:.6f}%")
    lines.append("")
    lines.append("=== Wrong-key decryption quality (from cipher_rgba.png) ===")
    lines.append(f"Wrong key: {WRONG_KEY}")
    lines.append(f"MAE_wrong:  {MAE_wrong}")
    lines.append(f"PSNR_wrong: {PSNR_wrong}")
    lines.append(f"SSIM_wrong: {SSIM_wrong}")
    lines.append("")
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
