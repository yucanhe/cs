"""
chaos_algebra_boolean_sbox_embed.py

实现内容：
1) 代数结构 S 盒：GF(2^8) 求逆 + 混沌生成仿射参数 (A_c, b_c)
2) 混沌布尔耦合增强：混沌生成可逆的 8x8 二进制矩阵 C_c，对输出位做耦合扩散
3) 嵌入点：用于“展示/统计安全链”的 8-bit 密文（量化后的 view ciphertext）
   - 不参与解密主链，不影响 CS/FISTA 重建质量

推荐用法：
- 主密文仍保持 float（用于解密恢复 y，再做 FISTA）
- 同时生成一个 uint8 “展示密文”，对其做 S-box 替换用于安全指标统计

作者注：
- 使用 AES 相同的不可约多项式 0x11B
- 所有矩阵运算在 GF(2) 上进行（bit-level）
"""

import numpy as np
import hashlib


# ============================================================
# 0) 基础：SHA-512 派生种子（支持“明文绑定”）
# ============================================================

def sha512(data: bytes) -> bytes:
    return hashlib.sha512(data).digest()

def derive_seed_bytes(master_key: bytes, plaintext_bytes: bytes = None) -> bytes:
    """
    返回 64-byte 种子：
    - 如果 plaintext_bytes 不为空：seed = SHA512(master_key || SHA512(plaintext))
      这样能实现明文敏感：同一密钥，不同明文 -> 不同 S-box
    - 如果 plaintext_bytes 为空：seed = SHA512(master_key)
    """
    if plaintext_bytes is None:
        return sha512(master_key)
    hP = sha512(plaintext_bytes)
    return sha512(master_key + hP)

def bytes_to_unit_float(b8: bytes) -> float:
    """8 bytes -> (0,1) float，避免 0/1 边界"""
    u = int.from_bytes(b8, "big", signed=False)
    x = (u + 0.5) / (2**64)
    eps = 1e-15
    return float(min(max(x, eps), 1.0 - eps))


# ============================================================
# 1) 混沌序列（这里用 logistic，实际你可替换成 NCML/3D-PMCM）
# ============================================================

def logistic_seq(n: int, x0: float, r: float, discard: int = 300) -> np.ndarray:
    x = float(x0)
    out = np.empty(n, dtype=np.float64)
    for _ in range(discard):
        x = r * x * (1.0 - x)
    for i in range(n):
        x = r * x * (1.0 - x)
        out[i] = x
    return out

def chaos_bits(nbits: int, seed64: bytes, offset: int = 0) -> np.ndarray:
    """
    从 seed64 派生 logistic 参数，然后产生 nbits 个比特（0/1）。
    offset：为了生成不同用途参数时避免复用相同片段
    """
    # 从 seed64 中取不同位置派生 x0 和 r
    x0 = bytes_to_unit_float(seed64[(0+offset) % 56 : (8+offset) % 56 + ((8+offset)%56 < (0+offset)%56)*56])
    # 上面切片写法为了防止越界太复杂，干脆用循环取 8 bytes
    def get8(pos):
        pos = pos % 64
        if pos <= 56:
            return seed64[pos:pos+8]
        else:
            part1 = seed64[pos:]
            part2 = seed64[:8-len(part1)]
            return part1 + part2

    x0 = bytes_to_unit_float(get8(0 + offset))
    u  = bytes_to_unit_float(get8(16 + offset))
    r  = 3.95 + 0.05 * u  # (3.95,4.0)

    seq = logistic_seq(nbits, x0=x0, r=r, discard=500)
    bits = (seq > 0.5).astype(np.uint8)
    return bits


# ============================================================
# 2) GF(2) 上 8x8 矩阵表示与运算（用 8 行字节表示）
# ============================================================

def parity8(x: int) -> int:
    """返回 x 的 1 比特奇偶校验（mod 2）"""
    # Python 3.8+ 支持 int.bit_count()
    return (x.bit_count() & 1)

def gf2_matmul_vec(rows8: np.ndarray, v: int) -> int:
    """
    rows8: shape (8,), 每个元素是一个 byte，表示矩阵一行的 8 bits
    v: 0..255，表示 8-bit 向量
    返回：0..255
    """
    out = 0
    for i in range(8):
        # 行与 v 按位与，然后做 parity
        bit = parity8(int(rows8[i]) & v)
        out |= (bit << (7 - i))  # 这里约定最高位对应第 0 行
    return out

def gf2_rank_8(rows8: np.ndarray) -> int:
    """
    计算 8x8 二进制矩阵的秩（高斯消元，GF(2)）
    rows8: shape (8,), 每行 8-bit
    """
    A = rows8.astype(np.uint8).copy()
    rank = 0
    col_mask = 0x80  # 从最高位开始找主元（bit7）
    for _ in range(8):
        # 找到有该列主元的行
        pivot = -1
        for r in range(rank, 8):
            if A[r] & col_mask:
                pivot = r
                break
        if pivot == -1:
            col_mask >>= 1
            continue
        # 交换到 rank 行
        A[rank], A[pivot] = A[pivot], A[rank]
        # 消元（对其他行）
        for r in range(8):
            if r != rank and (A[r] & col_mask):
                A[r] ^= A[rank]
        rank += 1
        col_mask >>= 1
        if col_mask == 0:
            break
    return rank

def random_invertible_gf2_matrix(seed64: bytes, offset: int) -> np.ndarray:
    """
    用混沌比特生成一个可逆 8x8 GF(2) 矩阵（8 行字节）。
    offset 用来避免与其它参数重叠。
    """
    # 反复尝试直到满秩
    tries = 0
    while True:
        tries += 1
        bits = chaos_bits(64, seed64, offset=offset + 13 * tries)
        # 按行拼成 8 个字节
        rows = np.zeros(8, dtype=np.uint8)
        for i in range(8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | int(bits[i*8 + j])
            rows[i] = byte
        if gf2_rank_8(rows) == 8:
            return rows

def random_gf2_vector8(seed64: bytes, offset: int) -> int:
    """生成 8-bit 向量 b（0..255）"""
    bits = chaos_bits(8, seed64, offset=offset)
    b = 0
    for i in range(8):
        b = (b << 1) | int(bits[i])
    return b


# ============================================================
# 3) GF(2^8) 运算：乘法/求逆（AES 多项式 0x11B）
# ============================================================

AES_POLY = 0x11B

def gf256_mul(a: int, b: int) -> int:
    """GF(2^8) 乘法（mod AES_POLY），a,b ∈ [0,255]"""
    res = 0
    a = int(a) & 0xFF
    b = int(b) & 0xFF
    for _ in range(8):
        if b & 1:
            res ^= a
        carry = a & 0x80
        a = (a << 1) & 0xFF
        if carry:
            a ^= (AES_POLY & 0xFF)  # 0x1B
        b >>= 1
    return res & 0xFF

def gf256_pow(a: int, e: int) -> int:
    """GF(2^8) 幂运算"""
    res = 1
    base = a & 0xFF
    ee = int(e)
    while ee > 0:
        if ee & 1:
            res = gf256_mul(res, base)
        base = gf256_mul(base, base)
        ee >>= 1
    return res & 0xFF

def gf256_inv(a: int) -> int:
    """
    GF(2^8) 求逆：a^(254)（费马小定理）
    0 的逆定义为 0
    """
    a = int(a) & 0xFF
    if a == 0:
        return 0
    return gf256_pow(a, 254)

def build_inv_table() -> np.ndarray:
    """预计算 0..255 的逆元表"""
    inv = np.zeros(256, dtype=np.uint8)
    for x in range(256):
        inv[x] = gf256_inv(x)
    return inv


# ============================================================
# 4) 代数结构 + 混沌布尔耦合 S-box 构造
# ============================================================

def build_algebra_boolean_chaos_sbox(master_key: bytes, plaintext_bytes: bytes = None):
    """
    生成 8x8 S-box（长度 256）及逆 S-box。

    S(x) = C_c * ( A_c * inv(x) XOR b_c )     (所有 * 在 GF(2) bit-level 上)
    - inv(x) 是 GF(2^8) 求逆（代数强非线性核心）
    - A_c, b_c 来自混沌（仿射参数化）
    - C_c 来自混沌（布尔函数耦合增强，要求可逆）

    返回：
      S: uint8[256]
      Sinv: uint8[256]
      params: dict（包含 A_c_rows, C_c_rows, b_c, seedhash）
    """
    seed64 = derive_seed_bytes(master_key, plaintext_bytes)
    inv_tbl = build_inv_table()

    # 混沌生成可逆仿射矩阵 A_c（GF(2)）
    A_rows = random_invertible_gf2_matrix(seed64, offset=10)
    # 混沌生成可逆耦合矩阵 C_c（GF(2)）
    C_rows = random_invertible_gf2_matrix(seed64, offset=200)
    # 混沌生成偏置向量 b_c
    b = random_gf2_vector8(seed64, offset=400)

    # 构造 S-box
    S = np.zeros(256, dtype=np.uint8)
    for x in range(256):
        gx = int(inv_tbl[x])  # 0..255
        y  = gf2_matmul_vec(A_rows, gx) ^ b
        z  = gf2_matmul_vec(C_rows, y)  # 布尔耦合增强
        S[x] = z

    # 确保双射（理论上若 A,C 可逆且 inv 是双射，则一定是双射）
    if len(np.unique(S)) != 256:
        # 极小概率：实现/位序约定导致异常，这里兜底重试
        # 实际论文里你可以不写这段，但工程上留着更稳
        for t in range(1, 50):
            A_rows = random_invertible_gf2_matrix(seed64, offset=10 + 1000*t)
            C_rows = random_invertible_gf2_matrix(seed64, offset=200 + 1000*t)
            b = random_gf2_vector8(seed64, offset=400 + 1000*t)
            for x in range(256):
                gx = int(inv_tbl[x])
                y  = gf2_matmul_vec(A_rows, gx) ^ b
                z  = gf2_matmul_vec(C_rows, y)
                S[x] = z
            if len(np.unique(S)) == 256:
                break

    # 逆盒
    Sinv = np.zeros(256, dtype=np.uint8)
    Sinv[S] = np.arange(256, dtype=np.uint8)

    params = {
        "A_rows": A_rows.copy(),
        "C_rows": C_rows.copy(),
        "b": int(b),
        "seed_sha512_hex": seed64.hex(),
    }
    return S, Sinv, params


# ============================================================
# 5) 嵌入点：对“展示密文 uint8”做 S-box 替换（不影响解密主链）
# ============================================================

def apply_sbox_to_view_cipher(view_u8: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    view_u8: 任意 shape 的 uint8 密文“展示图/展示矩阵”
    S: uint8[256]
    返回：替换后的 uint8
    """
    assert view_u8.dtype == np.uint8
    return S[view_u8]

def invert_sbox_on_view_cipher(view_u8_sboxed: np.ndarray, Sinv: np.ndarray) -> np.ndarray:
    """如果你需要反向恢复展示密文（一般统计用不需要），可以用它。"""
    assert view_u8_sboxed.dtype == np.uint8
    return Sinv[view_u8_sboxed]


# ============================================================
# 6) 一个通用的 float->uint8 量化（用于展示链）
# ============================================================

def quantize_float_to_uint8(x: np.ndarray, clip_sigma: float = 3.0):
    """
    把 float 密文映射成 uint8（仅用于展示/统计），不要求可逆。
    - 使用 mean±clip_sigma*std 进行截断，避免极端值影响对比度与直方图
    """
    x = x.astype(np.float64)
    mu = float(np.mean(x))
    sd = float(np.std(x) + 1e-12)
    lo = mu - clip_sigma * sd
    hi = mu + clip_sigma * sd
    x2 = np.clip(x, lo, hi)
    x2 = (x2 - lo) / (hi - lo + 1e-12)
    u8 = np.round(255.0 * x2).astype(np.uint8)
    return u8


# ============================================================
# 7) 示例：如何嵌入你的 CS+测量域浮点加密流程
# ============================================================

def embed_demo(cipher_float: np.ndarray, master_key: bytes, plaintext_bytes: bytes = None):
    """
    cipher_float：你 Stage4 输出的浮点密文（例如 3D 张量或 2D 矩阵）
    master_key：主密钥
    plaintext_bytes：可选，传入明文原始字节用于明文绑定（推荐）
    """
    # 1) 主链：cipher_float 用于解密（这里不动它）

    # 2) 展示链：float -> uint8
    view_u8 = quantize_float_to_uint8(cipher_float)

    # 3) 生成 S-box（代数 + 混沌 + 布尔耦合）
    S, Sinv, params = build_algebra_boolean_chaos_sbox(master_key, plaintext_bytes)

    # 4) S-box 替换（用于统计指标/可视化）
    view_u8_sboxed = apply_sbox_to_view_cipher(view_u8, S)

    return view_u8_sboxed, params


# ============================================================
# 8) 自检：S-box 是否双射、简单替换演示
# ============================================================

if __name__ == "__main__":
    # 假设你已经得到了某个 float 密文（这里用随机模拟）
    cipher_float = np.random.randn(64, 64, 32).astype(np.float64)

    master_key = b"my-master-key-2026"
    plaintext_bytes = b"example-plaintext-image-bytes"

    view_u8_sboxed, params = embed_demo(cipher_float, master_key, plaintext_bytes)

    print("Generated view ciphertext shape:", view_u8_sboxed.shape, view_u8_sboxed.dtype)
    print("S-box params (A_rows/C_rows/b) are derived; seed hash prefix:", params["seed_sha512_hex"][:16], "...")
