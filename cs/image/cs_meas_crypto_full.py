# cs_pipeline_spcmm_py38.py
# Python 3.8 compatible

import numpy as np
import hashlib
from typing import Optional, Tuple, Dict, Any

from skimage import io, color

try:
    from skimage.metrics import structural_similarity as ssim_metric
except Exception:
    ssim_metric = None


# ============================================================
# Stage 0: Key schedule / seed derivation
# ============================================================

def sha512(data: bytes) -> bytes:
    return hashlib.sha512(data).digest()

def derive_seed(master_key: bytes, plaintext_bytes: Optional[bytes]) -> bytes:
    """
    seed = SHA512(master_key || SHA512(plaintext))
    """
    if plaintext_bytes is None:
        return sha512(master_key)
    return sha512(master_key + sha512(plaintext_bytes))

def get8(seed: bytes, pos: int) -> bytes:
    pos = pos % 64
    if pos <= 56:
        return seed[pos:pos+8]
    part1 = seed[pos:]
    part2 = seed[:8-len(part1)]
    return part1 + part2

def bytes_to_unit_float(b8: bytes) -> float:
    """
    8 bytes -> (0,1) float，避开 0/1 边界
    """
    u = int.from_bytes(b8, "big", signed=False)
    x = (u + 0.5) / (2**64)
    eps = 1e-15
    if x < eps:
        x = eps
    if x > 1.0 - eps:
        x = 1.0 - eps
    return float(x)


# ============================================================
# Chaos source: your nD-SPCMM
# ============================================================

def nD_spcmm(n: int,
            steps: int,
            x0,
            a,
            b,
            w,
            c,
            theta: float = 1.0,
            dtype=np.float64) -> np.ndarray:
    """
    n维正弦-多项式复合模映射 (nD-SPCMM)
    输出范围 [0, theta)
    """
    x_history = np.zeros((steps, n), dtype=dtype)
    x_history[0] = np.array(x0, dtype=dtype)

    a = np.array(a, dtype=dtype)
    b = np.array(b, dtype=dtype)
    w = np.array(w, dtype=dtype)
    c = dtype(c)
    theta = dtype(theta)

    for i in range(steps - 1):
        curr = x_history[i]
        nxt = np.zeros(n, dtype=dtype)
        for k in range(n - 1, -1, -1):
            term = a[k] * np.sin(b[k] * curr[k])
            if k < n - 1:
                coupling = np.sum(w[k+1:] * (curr[k+1:] ** 2))
            else:
                coupling = dtype(0.0)
            nxt[k] = (term + coupling + c) % theta
        x_history[i + 1] = nxt
    return x_history


def spcmm_params_from_seed(seed: bytes, n_dim: int, stream_id: int):
    """
    用 seed 派生 nD-SPCMM 的参数（可复现，可分流）
    """
    base = 8 * (stream_id * 7)

    # x0 in (0,1)
    x0 = np.array([bytes_to_unit_float(get8(seed, base + 8*i))
                   for i in range(n_dim)], dtype=np.float64)

    # a in [2.8, 4.0]
    a = np.array([2.8 + 1.2 * bytes_to_unit_float(get8(seed, base + 8*(n_dim+i)))
                  for i in range(n_dim)], dtype=np.float64)

    # b in [6, 16]
    b = np.array([6.0 + 10.0 * bytes_to_unit_float(get8(seed, base + 8*(2*n_dim+i)))
                  for i in range(n_dim)], dtype=np.float64)

    # w: w[0]=0, others in [0.2, 1.0]
    w = np.array([0.0] + [0.2 + 0.8 * bytes_to_unit_float(get8(seed, base + 8*(3*n_dim+i)))
                          for i in range(1, n_dim)], dtype=np.float64)

    # c in [0,1)
    c = bytes_to_unit_float(get8(seed, base + 8*(4*n_dim + 0)))

    theta = 1.0
    return x0, a, b, w, c, theta


def spcmm_uniform(seed: bytes,
                  length: int,
                  offset: int,
                  n_dim: int = 3,
                  discard: int = 1200) -> np.ndarray:
    """
    统一混沌源：输出 (0,1) 序列
    offset -> stream_id 分流
    """
    stream_id = int(offset) % 997  # 素数分流
    x0, a, b, w, c, theta = spcmm_params_from_seed(seed, n_dim=n_dim, stream_id=stream_id)

    steps = discard + length
    traj = nD_spcmm(n_dim, steps, x0, a, b, w, c, theta=theta, dtype=np.float64)

    seq = traj[discard:, 0].astype(np.float64)  # 取第0维

    eps = 1e-15
    seq = np.clip(seq, eps, 1.0 - eps)
    return seq


def chaos_perm(seed: bytes, n: int, offset: int) -> np.ndarray:
    """
    索引排序法 -> 置乱perm
    """
    seq = spcmm_uniform(seed, n, offset=offset, n_dim=3)
    idx = np.arange(n, dtype=np.int32)
    perm = np.lexsort((idx, seq)).astype(np.int32)
    return perm

def inv_perm(perm: np.ndarray) -> np.ndarray:
    inv = np.zeros_like(perm)
    inv[perm] = np.arange(len(perm), dtype=perm.dtype)
    return inv


# ============================================================
# Stage 1: IO + normalize
# ============================================================

def load_grayscale_normalized(path: str) -> np.ndarray:
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = img.astype(np.float64)
    img01 = (img - img.min()) / (img.max() - img.min() + 1e-12)
    return img01

def to_uint8(img01: np.ndarray) -> np.ndarray:
    return np.round(np.clip(img01, 0, 1) * 255.0).astype(np.uint8)


# ============================================================
# Stage 2: Blockify / Unblockify
# ============================================================

def pad_to_block(img01: np.ndarray, B: int, mode: str = "edge"):
    H, W = img01.shape
    Hp = int(np.ceil(H / B) * B)
    Wp = int(np.ceil(W / B) * B)
    pad_h = Hp - H
    pad_w = Wp - W
    if pad_h == 0 and pad_w == 0:
        return img01, (H, W)
    img_pad = np.pad(img01, ((0, pad_h), (0, pad_w)), mode=mode)
    return img_pad, (H, W)

def blockify(img01: np.ndarray, B: int) -> np.ndarray:
    H, W = img01.shape
    Hb, Wb = H // B, W // B
    return img01.reshape(Hb, B, Wb, B).transpose(0, 2, 1, 3).copy()

def unblockify(blocks: np.ndarray) -> np.ndarray:
    Hb, Wb, B, _ = blocks.shape
    return blocks.transpose(0, 2, 1, 3).reshape(Hb * B, Wb * B)


# ============================================================
# Stage 3: DCT / IDCT
# ============================================================

def dct_matrix(N: int) -> np.ndarray:
    C = np.zeros((N, N), dtype=np.float64)
    factor = np.pi / N
    for k in range(N):
        alpha = np.sqrt(1.0 / N) if k == 0 else np.sqrt(2.0 / N)
        for n in range(N):
            C[k, n] = alpha * np.cos((n + 0.5) * k * factor)
    return C

def dct2(block: np.ndarray, C: np.ndarray) -> np.ndarray:
    return C @ block @ C.T

def idct2(coef: np.ndarray, C: np.ndarray) -> np.ndarray:
    return C.T @ coef @ C

def dct_blocks(blocks: np.ndarray, C: np.ndarray) -> np.ndarray:
    Hb, Wb, B, _ = blocks.shape
    out = np.empty_like(blocks, dtype=np.float64)
    for i in range(Hb):
        for j in range(Wb):
            out[i, j] = dct2(blocks[i, j], C)
    return out

def idct_blocks(coefs: np.ndarray, C: np.ndarray) -> np.ndarray:
    Hb, Wb, B, _ = coefs.shape
    out = np.empty_like(coefs, dtype=np.float64)
    for i in range(Hb):
        for j in range(Wb):
            out[i, j] = idct2(coefs[i, j], C)
    return out


# ============================================================
# Stage 4: Sparse-domain permutations
# ============================================================

def intra_block_permute_coefs(coefs: np.ndarray, perm_intra: np.ndarray) -> np.ndarray:
    Hb, Wb, B, _ = coefs.shape
    n = B * B
    out = np.empty_like(coefs, dtype=np.float64)
    for i in range(Hb):
        for j in range(Wb):
            v = coefs[i, j].reshape(n)
            out[i, j] = v[perm_intra].reshape(B, B)
    return out

def intra_block_inverse_permute_coefs(coefs_perm: np.ndarray, inv_perm_intra: np.ndarray) -> np.ndarray:
    Hb, Wb, B, _ = coefs_perm.shape
    n = B * B
    out = np.empty_like(coefs_perm, dtype=np.float64)
    for i in range(Hb):
        for j in range(Wb):
            v = coefs_perm[i, j].reshape(n)
            out[i, j] = v[inv_perm_intra].reshape(B, B)
    return out

def inter_block_permute(arr: np.ndarray, perm_inter: np.ndarray) -> np.ndarray:
    Hb, Wb = arr.shape[:2]
    rest = arr.shape[2:]
    flat = arr.reshape(Hb * Wb, *rest)
    flat_p = flat[perm_inter]
    return flat_p.reshape(Hb, Wb, *rest)

def inter_block_inverse_permute(arr_perm: np.ndarray, inv_perm_inter: np.ndarray) -> np.ndarray:
    Hb, Wb = arr_perm.shape[:2]
    rest = arr_perm.shape[2:]
    flat = arr_perm.reshape(Hb * Wb, *rest)
    flat_r = flat[inv_perm_inter]
    return flat_r.reshape(Hb, Wb, *rest)


# ============================================================
# Stage 5: CS measurement matrix + measurement
# ============================================================

def build_measurement_matrix(seed: bytes, m: int, n: int, offset: int) -> np.ndarray:
    seq = spcmm_uniform(seed, m * n, offset=offset, n_dim=3)
    A = seq.reshape(m, n).astype(np.float64)
    A = (A - 0.5) / 0.5  # [-1,1]
    col_norm = np.linalg.norm(A, axis=0, keepdims=True) + 1e-12
    A = A / col_norm
    return A

def measure_blocks(coefs: np.ndarray, A: np.ndarray) -> np.ndarray:
    Hb, Wb, B, _ = coefs.shape
    n = B * B
    m = A.shape[0]
    Y = np.zeros((Hb, Wb, m), dtype=np.float64)
    for i in range(Hb):
        for j in range(Wb):
            s = coefs[i, j].reshape(n)
            Y[i, j] = A @ s
    return Y


# ============================================================
# Stage 6: Measurement-domain 3D cyclic shift (invertible)
# ============================================================

def compute_shifts(seed: bytes, Hb: int, Wb: int, M: int, offset: int) -> Tuple[int, int, int]:
    u = spcmm_uniform(seed, 3, offset=offset, n_dim=3)
    sh_h = int(np.floor(u[0] * Hb)) if Hb > 0 else 0
    sh_w = int(np.floor(u[1] * Wb)) if Wb > 0 else 0
    sh_m = int(np.floor(u[2] * M))  if M  > 0 else 0
    return sh_h, sh_w, sh_m

def roll3d(X: np.ndarray, sh_h: int, sh_w: int, sh_m: int) -> np.ndarray:
    out = np.roll(X, sh_h, axis=0)
    out = np.roll(out, sh_w, axis=1)
    out = np.roll(out, sh_m, axis=2)
    return out

def inv_roll3d(X: np.ndarray, sh_h: int, sh_w: int, sh_m: int) -> np.ndarray:
    out = np.roll(X, -sh_h, axis=0)
    out = np.roll(out, -sh_w, axis=1)
    out = np.roll(out, -sh_m, axis=2)
    return out


# ============================================================
# Stage 7: Measurement-domain reversible float diffusion
# ============================================================

def build_mask_R(seed: bytes, shape3d: Tuple[int, int, int], offset: int) -> np.ndarray:
    Hb, Wb, M = shape3d
    seq = spcmm_uniform(seed, Hb * Wb * M, offset=offset, n_dim=3)
    R = (seq - 0.5) / 0.5  # [-1,1]
    return R.reshape(Hb, Wb, M).astype(np.float64)

def diffusion_forward(Y: np.ndarray, R: np.ndarray, beta: float, eps: float) -> np.ndarray:
    Hb, Wb, M = Y.shape
    C = np.zeros_like(Y, dtype=np.float64)
    for i in range(Hb):
        for j in range(Wb):
            for k in range(M):
                prev = 0.0
                if i > 0: prev += C[i-1, j, k]
                if j > 0: prev += C[i, j-1, k]
                if k > 0: prev += C[i, j, k-1]
                C[i, j, k] = Y[i, j, k] + beta * R[i, j, k] + eps * prev
    return C

def diffusion_inverse(C: np.ndarray, R: np.ndarray, beta: float, eps: float) -> np.ndarray:
    Hb, Wb, M = C.shape
    Y = np.zeros_like(C, dtype=np.float64)
    for i in range(Hb):
        for j in range(Wb):
            for k in range(M):
                prev = 0.0
                if i > 0: prev += C[i-1, j, k]
                if j > 0: prev += C[i, j-1, k]
                if k > 0: prev += C[i, j, k-1]
                Y[i, j, k] = C[i, j, k] - beta * R[i, j, k] - eps * prev
    return Y


# ============================================================
# Stage 8: FISTA reconstruct per block
# ============================================================

def soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)

def fista(A: np.ndarray, y: np.ndarray, lam: float, max_iter: int) -> np.ndarray:
    x = np.zeros(A.shape[1], dtype=np.float64)
    z = x.copy()
    t = 1.0
    L = np.linalg.norm(A.T @ A, 2)
    step = 1.0 / (L + 1e-12)

    for _ in range(max_iter):
        grad = A.T @ (A @ z - y)
        x_new = soft_threshold(z - step * grad, lam * step)
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        z = x_new + ((t - 1) / t_new) * (x_new - x)
        x, t = x_new, t_new
    return x

def reconstruct_coefs_from_measurements(Y: np.ndarray, A: np.ndarray, B: int, lam: float, max_iter: int) -> np.ndarray:
    Hb, Wb, _ = Y.shape
    coefs = np.zeros((Hb, Wb, B, B), dtype=np.float64)
    for i in range(Hb):
        for j in range(Wb):
            s_hat = fista(A, Y[i, j], lam=lam, max_iter=max_iter)
            coefs[i, j] = s_hat.reshape(B, B)
    return coefs


# ============================================================
# Stage 9: Quality metrics
# ============================================================

def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 99.0
    return 10 * np.log10(1.0 / mse)

def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    if ssim_metric is None:
        return float("nan")
    return float(ssim_metric(img1, img2, data_range=1.0))


# ============================================================
# End-to-end: encrypt / decrypt
# ============================================================

def encrypt_pipeline(img01: np.ndarray,
                     master_key: bytes,
                     B: int = 8,
                     m_rate: float = 0.6,
                     beta: float = 0.15,
                     eps: float = 0.25):
    img_pad, orig_shape = pad_to_block(img01, B, mode="edge")
    blocks = blockify(img_pad, B)
    Hb, Wb = blocks.shape[:2]
    n = B * B
    m = int(np.round(m_rate * n))
    m = max(1, min(m, n))

    # 明文绑定 seed（工程可行；论文里写“参数安全封装/会话密钥”）
    img_u8 = to_uint8(img_pad)
    seed = derive_seed(master_key, img_u8.tobytes())

    # DCT
    C = dct_matrix(B)
    coefs = dct_blocks(blocks, C)

    # intra perm
    p_intra = chaos_perm(seed, n, offset=50)
    ip_intra = inv_perm(p_intra)
    coefs_intra = intra_block_permute_coefs(coefs, p_intra)

    # inter perm
    total_blocks = Hb * Wb
    p_inter = chaos_perm(seed, total_blocks, offset=80)
    ip_inter = inv_perm(p_inter)
    coefs_perm = inter_block_permute(coefs_intra, p_inter)

    # CS measurement
    A = build_measurement_matrix(seed, m=m, n=n, offset=120)
    Y = measure_blocks(coefs_perm, A)

    # 3D shift
    sh = compute_shifts(seed, Hb, Wb, m, offset=200)
    Y_shift = roll3d(Y, sh[0], sh[1], sh[2])

    # diffusion
    R = build_mask_R(seed, (Hb, Wb, m), offset=260)
    C_float = diffusion_forward(Y_shift, R, beta=beta, eps=eps)

    meta = {
        "orig_shape": orig_shape,
        "pad_shape": img_pad.shape,
        "B": B,
        "Hb": Hb,
        "Wb": Wb,
        "n": n,
        "m": m,
        "seed_hex": seed.hex(),
        "A": A,
        "C_dct": C,
        "inv_perm_intra": ip_intra,
        "inv_perm_inter": ip_inter,
        "shift": sh,
        "beta": beta,
        "eps": eps,
    }
    return C_float, meta


def decrypt_pipeline(C_float: np.ndarray,
                     meta: Dict[str, Any],
                     master_key: bytes,
                     lam: float = 0.005,
                     max_iter: int = 400) -> np.ndarray:
    B = meta["B"]
    Hb, Wb = meta["Hb"], meta["Wb"]
    m = meta["m"]
    A = meta["A"]
    C_dct = meta["C_dct"]
    ip_intra = meta["inv_perm_intra"]
    ip_inter = meta["inv_perm_inter"]
    sh = meta["shift"]
    beta, eps = meta["beta"], meta["eps"]

    seed = bytes.fromhex(meta["seed_hex"])

    # inverse diffusion
    R = build_mask_R(seed, (Hb, Wb, m), offset=260)
    Y_shift = diffusion_inverse(C_float, R, beta=beta, eps=eps)

    # inverse shift
    Y = inv_roll3d(Y_shift, sh[0], sh[1], sh[2])

    # inverse inter perm (measurement domain)
    Y_restored = inter_block_inverse_permute(Y, ip_inter)

    # reconstruct coefs (intra-permuted)
    coefs_perm = reconstruct_coefs_from_measurements(Y_restored, A, B, lam=lam, max_iter=max_iter)

    # inverse intra perm
    coefs_rec = intra_block_inverse_permute_coefs(coefs_perm, ip_intra)

    # IDCT
    blocks_rec = idct_blocks(coefs_rec, C_dct)
    img_pad_rec = unblockify(blocks_rec)
    img_pad_rec = np.clip(img_pad_rec, 0.0, 1.0)

    # crop
    H0, W0 = meta["orig_shape"]
    return img_pad_rec[:H0, :W0]


def main():
    img_path = "1.png"
    master_key = b"my-master-key-2026"

    img01 = load_grayscale_normalized(img_path)

    C_float, meta = encrypt_pipeline(img01, master_key, B=8, m_rate=0.6, beta=0.15, eps=0.25)
    np.savez("cipher_float_meta_spcmm_py38.npz", C_float=C_float, meta=meta)

    img_rec = decrypt_pipeline(C_float, meta, master_key, lam=0.005, max_iter=400)

    print("PSNR:", psnr(img01, img_rec))
    print("SSIM:", ssim(img01, img_rec))

    io.imsave("decrypted_spcmm_py38.png", to_uint8(img_rec))


if __name__ == "__main__":
    main()
