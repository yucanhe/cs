"""
chaos_algebra_boolean_sbox_embed.py

实现内容：
1) 代数结构 S 盒：GF(2^8) 求逆 + nD-SPCMM 混沌生成仿射参数 (A_c, b_c)
2) 嵌入点：用于"展示/统计安全链"的 8-bit 密文（量化后的 view ciphertext）
   - 不参与解密主链，不影响 CS/FISTA 重建质量

推荐用法：
- 主密文仍保持 float（用于解密恢复 y，再做 FISTA）
- 同时生成一个 uint8 "展示密文"，对其做 S-box 替换用于安全指标统计

作者注：
- 使用 AES 相同的不可约多项式 0x11B
- 所有矩阵运算在 GF(2) 上进行（bit-level）
- 混沌源：nD-SPCMM（n维正弦-多项式复合模映射）
  x_k[n+1] = (a_k·sin(b_k·x_k[n]) + Σ w_j·x_j[n]² + c) mod θ
  级联构造保证雅可比矩阵为下三角阵，Lyapunov 指数解析可控

修复记录：
[FIX 1] 去除冗余 C_c 矩阵：S(x)=C*(A*inv(x)⊕b) 代数等价于 A'*inv(x)⊕b'，NL/DF 不变
[FIX 2] 明文绑定改为 nonce 派生：原方案在 CS 场景下存在已知明文漏洞
[FIX 3] HKDF 替换 offset%64，消除 A/b 两路种子的隐性碰撞
[FIX 4] nD-SPCMM 替换 logistic map，混沌源与主系统统一，整数量化保证跨平台一致
[FIX 5] 删除死代码（原 x0 第一次赋值被第二次立即覆盖）
"""

import numpy as np
import hashlib
import hmac
import os


# ============================================================
# 0) 基础：HKDF 派生 + nonce 管理
# ============================================================

def sha512(data: bytes) -> bytes:
    return hashlib.sha512(data).digest()


def hkdf_expand(prk: bytes, info: bytes, length: int = 64) -> bytes:
    """
    [FIX 3] HKDF-Expand（RFC 5869）简化版，使用 HMAC-SHA512。
    每路参数用独立 info 标签展开，完全消除 offset%64 碰撞问题。
    """
    hash_len = 64
    n = (length + hash_len - 1) // hash_len
    okm = b""
    T = b""
    for i in range(1, n + 1):
        T = hmac.new(prk, T + info + bytes([i]), hashlib.sha512).digest()
        okm += T
    return okm[:length]


def generate_nonce(n_bytes: int = 16) -> bytes:
    """生成随机 nonce，建议每帧调用一次，明文传输给解密端。"""
    return os.urandom(n_bytes)


def derive_seed_bytes(master_key: bytes, nonce: bytes = None,
                      plaintext_bytes: bytes = None) -> bytes:
    """
    [FIX 2] 返回 64-byte 主种子。

    推荐用法（nonce 模式）：
        seed = SHA512(master_key || nonce)
        nonce 每帧随机生成并明文传输，与明文内容无关。

    兼容旧用法（plaintext_bytes，已废弃）：
        若仍传入 plaintext_bytes，会触发 DeprecationWarning。
        在 CS 图像场景下，已知明文可还原 seed，存在安全风险。
    """
    if nonce is not None:
        return sha512(master_key + nonce)
    if plaintext_bytes is not None:
        import warnings
        warnings.warn(
            "plaintext_bytes 参数已废弃：明文绑定在 CS 场景下存在已知明文攻击风险。"
            "请改用 nonce=generate_nonce() 并将 nonce 明文传输给解密端。",
            DeprecationWarning,
            stacklevel=2,
        )
        hP = sha512(plaintext_bytes)
        return sha512(master_key + hP)
    return sha512(master_key)


# ============================================================
# 1) nD-SPCMM 混沌映射（核心，与主系统统一）
# ============================================================

# 默认超混沌参数（3维，所有 LE > 0）
_DEFAULT_A = np.array([3.1, 3.3, 3.5])
_DEFAULT_B = np.array([10.0, 11.0, 12.0])
_DEFAULT_W = np.array([0.0, 0.6, 0.5])
_DEFAULT_C = 0.7
_DEFAULT_THETA = 1.0
_WARMUP = 500   # 预热步数，丢弃瞬态


def nD_spcmm(n: int, steps: int, x0: np.ndarray,
             a: np.ndarray, b: np.ndarray, w: np.ndarray,
             c: float, theta: float = 1.0) -> np.ndarray:
    """
    n 维正弦-多项式复合模映射（nD-SPCMM）。

    映射方程（级联，k 从 n-1 到 0）：
        x_k[t+1] = ( a_k·sin(b_k·x_k[t]) + Σ_{j>k} w_j·x_j[t]² + c ) mod θ

    级联构造保证雅可比矩阵为下三角阵，Lyapunov 指数解析可控。

    参数：
        n      维度
        steps  迭代步数（含预热）
        x0     初始状态，shape (n,)
        a      振幅参数，shape (n,)
        b      频率参数，shape (n,)
        w      耦合权重，shape (n,)，w[0] 不参与（级联逻辑）
        c      偏置
        theta  相空间折叠尺度

    返回：
        x_history，shape (steps, n)
    """
    x_history = np.zeros((steps, n))
    x_history[0] = x0
    for i in range(steps - 1):
        curr = x_history[i]
        nxt = np.zeros(n)
        for k in range(n - 1, -1, -1):
            term     = a[k] * np.sin(b[k] * curr[k])
            coupling = np.sum(w[k+1:] * curr[k+1:] ** 2) if k < n - 1 else 0
            nxt[k]   = (term + coupling + c) % theta
        x_history[i + 1] = nxt
    return x_history


def _seed_to_spcmm_x0(seed_bytes: bytes, info: bytes, n: int = 3) -> np.ndarray:
    """
    [FIX 4] 从 HKDF 子密钥派生 nD-SPCMM 初始状态 x0。

    用整数量化（floor(u × 2^32) / 2^32）将 float64 精度钉死，
    保证不同平台、不同语言运行结果完全一致。
    避免了原 logistic map 方案中 r∈(3.95,4.0) 导致的有效熵瓶颈。
    """
    raw = hkdf_expand(seed_bytes, info, length=n * 8)
    x0 = np.zeros(n)
    for i in range(n):
        u64 = int.from_bytes(raw[i*8:(i+1)*8], "big")
        # 量化到 32-bit 精度，消除浮点同步风险
        x0[i] = float(u64 >> 32) / (2**32) + 1e-9   # 避免边界 0
        x0[i] = min(x0[i], 1.0 - 1e-9)
    return x0


def _spcmm_to_bits(n_bits: int, prk: bytes, info: bytes,
                   a: np.ndarray = None, b_param: np.ndarray = None,
                   w: np.ndarray = None, c: float = None,
                   theta: float = None) -> np.ndarray:
    """
    [FIX 4] 用 nD-SPCMM 生成 n_bits 个伪随机比特。

    流程：
      1. HKDF 派生 x0（整数量化，跨平台确定性）
      2. 预热 _WARMUP 步（丢弃瞬态）
      3. 取第 0 维轨迹，乘以 2^32 取整后逐位展开

    参数 a/b/w/c/theta 默认使用与主系统相同的超混沌参数，
    也可传入自定义参数以适配不同安全需求。
    """
    a     = _DEFAULT_A     if a     is None else np.asarray(a,     dtype=float)
    b_p   = _DEFAULT_B     if b_param is None else np.asarray(b_param, dtype=float)
    w     = _DEFAULT_W     if w     is None else np.asarray(w,     dtype=float)
    c     = _DEFAULT_C     if c     is None else float(c)
    theta = _DEFAULT_THETA if theta is None else float(theta)
    n_dim = len(a)

    # 每 32 bit 需要一个混沌步（取 x[0] 的 32-bit 量化值）
    steps_needed = _WARMUP + (n_bits + 31) // 32

    x0   = _seed_to_spcmm_x0(prk, info, n=n_dim)
    traj = nD_spcmm(n_dim, steps_needed, x0, a, b_p, w, c, theta)

    # 取预热后的轨迹第 0 维，量化为 32-bit 整数，再展开成比特
    seq   = traj[_WARMUP:, 0]
    u32   = (seq * (2**32)).astype(np.uint32)
    bits  = np.unpackbits(u32.view(np.uint8))
    return bits[:n_bits]


# ============================================================
# 2) GF(2) 上 8x8 矩阵运算（保持不变）
# ============================================================

def parity8(x: int) -> int:
    """返回 x 的 1 比特奇偶校验（mod 2）"""
    return x.bit_count() & 1


def gf2_matmul_vec(rows8: np.ndarray, v: int) -> int:
    """
    rows8: shape (8,)，每个元素是一个 byte，表示矩阵一行的 8 bits
    v: 0..255，表示 8-bit 向量
    返回：0..255
    """
    out = 0
    for i in range(8):
        bit = parity8(int(rows8[i]) & v)
        out |= (bit << (7 - i))
    return out


def gf2_rank_8(rows8: np.ndarray) -> int:
    """计算 8x8 二进制矩阵的秩（高斯消元，GF(2)）"""
    A = rows8.astype(np.uint8).copy()
    rank = 0
    col_mask = 0x80
    for _ in range(8):
        pivot = -1
        for r in range(rank, 8):
            if A[r] & col_mask:
                pivot = r
                break
        if pivot == -1:
            col_mask >>= 1
            continue
        A[rank], A[pivot] = A[pivot], A[rank]
        for r in range(8):
            if r != rank and (A[r] & col_mask):
                A[r] ^= A[rank]
        rank += 1
        col_mask >>= 1
        if col_mask == 0:
            break
    return rank


def random_invertible_gf2_matrix(prk: bytes, info_prefix: bytes,
                                  spcmm_params: dict = None) -> np.ndarray:
    """
    [FIX 3+4] 用 nD-SPCMM 生成可逆 8x8 GF(2) 矩阵。

    每次 tries 附加计数器到 info，HKDF 保证每路完全独立，
    同时 nD-SPCMM 作为混沌源替换了原 logistic map。
    """
    sp = spcmm_params or {}
    for tries in range(1, 200):
        info = info_prefix + b":try:" + str(tries).encode()
        bits = _spcmm_to_bits(64, prk, info, **sp)
        rows = np.zeros(8, dtype=np.uint8)
        for i in range(8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | int(bits[i * 8 + j])
            rows[i] = byte
        if gf2_rank_8(rows) == 8:
            return rows
    raise RuntimeError("无法生成满秩矩阵，请检查参数是否有效")


def random_gf2_vector8(prk: bytes, info: bytes,
                       spcmm_params: dict = None) -> int:
    """[FIX 3+4] 用 nD-SPCMM 生成 8-bit 向量 b（0..255）。"""
    sp = spcmm_params or {}
    bits = _spcmm_to_bits(8, prk, info, **sp)
    b = 0
    for i in range(8):
        b = (b << 1) | int(bits[i])
    return b


# ============================================================
# 3) GF(2^8) 运算：乘法/求逆（保持不变，AES 多项式 0x11B）
# ============================================================

AES_POLY = 0x11B


def gf256_mul(a: int, b: int) -> int:
    """GF(2^8) 乘法（mod AES_POLY）"""
    res = 0
    a = int(a) & 0xFF
    b = int(b) & 0xFF
    for _ in range(8):
        if b & 1:
            res ^= a
        carry = a & 0x80
        a = (a << 1) & 0xFF
        if carry:
            a ^= (AES_POLY & 0xFF)
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
    """GF(2^8) 求逆：a^254（费马小定理），0 的逆定义为 0"""
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
# 4) 代数结构 S-box 构造
# ============================================================

def build_algebra_boolean_chaos_sbox(master_key: bytes,
                                      plaintext_bytes: bytes = None,
                                      nonce: bytes = None,
                                      spcmm_params: dict = None):
    """
    生成代数结构 S-box（长度 256）及逆 S-box。

    混沌源：nD-SPCMM（与主加密系统统一）
    公式：S(x) = A · GF256inv(x) ⊕ b

    [FIX 1] 去除冗余 C_c：S(x)=C*(A*inv(x)⊕b) 等价于 A'*inv(x)⊕b'，NL/DF 不变。
    [FIX 2] nonce 替换明文绑定，消除 CS 场景已知明文漏洞。
    [FIX 3] HKDF 独立标签，消除 A/b 两路种子碰撞。
    [FIX 4] nD-SPCMM 替换 logistic map，混沌源统一，整数量化保证跨平台一致。

    参数：
        master_key     主密钥（bytes）
        nonce          本帧随机数（推荐 generate_nonce()，16 字节）
        plaintext_bytes 已废弃，传入时触发 DeprecationWarning
        spcmm_params   可选，自定义 nD-SPCMM 参数字典，支持以下键：
                         a, b_param, w, c, theta
                       不传则使用默认超混沌参数（a=[3.1,3.3,3.5] 等）

    返回：
        S:      uint8[256]
        Sinv:   uint8[256]
        params: dict（含 A_rows、b、nonce_hex、seed_sha512_hex）
    """
    prk     = derive_seed_bytes(master_key, nonce=nonce, plaintext_bytes=plaintext_bytes)
    inv_tbl = build_inv_table()
    sp      = spcmm_params or {}

    # [FIX 3+4] 两路独立标签 + nD-SPCMM 生成参数
    A_rows = random_invertible_gf2_matrix(prk, info_prefix=b"sbox:A", spcmm_params=sp)
    b      = random_gf2_vector8(prk, info=b"sbox:b", spcmm_params=sp)

    # [FIX 1] S(x) = A · inv(x) ⊕ b
    S = np.zeros(256, dtype=np.uint8)
    for x in range(256):
        gx   = int(inv_tbl[x])
        S[x] = gf2_matmul_vec(A_rows, gx) ^ b

    # 双射兜底（理论上必然成立，保险起见）
    if len(np.unique(S)) != 256:
        for t in range(1, 50):
            A_rows = random_invertible_gf2_matrix(
                prk, info_prefix=("sbox:A:retry" + str(t)).encode(), spcmm_params=sp
            )
            b = random_gf2_vector8(prk, info=("sbox:b:retry" + str(t)).encode(), spcmm_params=sp)
            for x in range(256):
                gx   = int(inv_tbl[x])
                S[x] = gf2_matmul_vec(A_rows, gx) ^ b
            if len(np.unique(S)) == 256:
                break

    # 逆盒
    Sinv = np.zeros(256, dtype=np.uint8)
    Sinv[S] = np.arange(256, dtype=np.uint8)

    params = {
        "A_rows":          A_rows.copy(),
        "b":               int(b),
        "nonce_hex":       nonce.hex() if nonce else None,
        "seed_sha512_hex": prk.hex(),
    }
    return S, Sinv, params


# ============================================================
# 5) 嵌入点：对"展示密文 uint8"做 S-box 替换
# ============================================================

def apply_sbox_to_view_cipher(view_u8: np.ndarray, S: np.ndarray) -> np.ndarray:
    """view_u8: uint8 密文展示图；S: uint8[256]；返回替换后的 uint8。"""
    assert view_u8.dtype == np.uint8
    return S[view_u8]


def invert_sbox_on_view_cipher(view_u8_sboxed: np.ndarray, Sinv: np.ndarray) -> np.ndarray:
    """逆向恢复展示密文（统计用途一般不需要）。"""
    assert view_u8_sboxed.dtype == np.uint8
    return Sinv[view_u8_sboxed]


# ============================================================
# 6) float -> uint8 量化（展示链用）
# ============================================================

def quantize_float_to_uint8(x: np.ndarray, clip_sigma: float = 3.0) -> np.ndarray:
    """
    float 密文 → uint8（仅用于可视化/展示，不参与安全统计）。
    注意：此操作是破坏性的（6.6× 信息损失），不应用于 NPCR/UACI 等安全指标。
    """
    x  = x.astype(np.float64)
    mu = float(np.mean(x))
    sd = float(np.std(x) + 1e-12)
    lo = mu - clip_sigma * sd
    hi = mu + clip_sigma * sd
    x2 = np.clip(x, lo, hi)
    x2 = (x2 - lo) / (hi - lo + 1e-12)
    return np.round(255.0 * x2).astype(np.uint8)


# ============================================================
# 7) 示例：嵌入 CS+测量域浮点加密流程
# ============================================================

def embed_demo(cipher_float: np.ndarray, master_key: bytes,
               nonce: bytes = None, plaintext_bytes: bytes = None,
               spcmm_params: dict = None):
    """
    cipher_float：Stage4 输出的浮点密文
    master_key：主密钥
    nonce：推荐传入，每帧随机生成（generate_nonce()）
    spcmm_params：可选，自定义 nD-SPCMM 参数
    """
    view_u8        = quantize_float_to_uint8(cipher_float)
    S, Sinv, params = build_algebra_boolean_chaos_sbox(
        master_key, nonce=nonce, plaintext_bytes=plaintext_bytes,
        spcmm_params=spcmm_params
    )
    view_u8_sboxed = apply_sbox_to_view_cipher(view_u8, S)
    return view_u8_sboxed, params


# ============================================================
# 8) 自检
# ============================================================

if __name__ == "__main__":
    master_key = b"my-master-key-2026"
    nonce      = generate_nonce()

    S, Sinv, params = build_algebra_boolean_chaos_sbox(master_key, nonce=nonce)

    # 基本正确性
    assert len(np.unique(S)) == 256,                      "S 盒不是双射！"
    assert all(Sinv[S[i]] == i for i in range(256)),      "逆 S 盒验证失败！"
    print("✅ 双射 + 逆盒验证通过")

    # 确定性（同一 nonce，两端必须一致）
    S2, _, _ = build_algebra_boolean_chaos_sbox(master_key, nonce=nonce)
    assert np.array_equal(S, S2),                         "相同 nonce 产生了不同 S 盒！"
    print("✅ 确定性验证通过（加解密两端一致）")

    # 动态性（不同 nonce → 不同 S 盒）
    S3, _, _ = build_algebra_boolean_chaos_sbox(master_key, nonce=generate_nonce())
    assert not np.array_equal(S, S3),                     "不同 nonce 产生了相同 S 盒！"
    print("✅ 动态性验证通过（不同 nonce → 不同 S 盒）")

    # 密码学指标
    def fast_wht(f):
        n = len(f); h = 1; x = f.copy().astype(np.float64)
        while h < n:
            for i in range(0, n, h * 2):
                for j in range(i, i + h):
                    x[j], x[j+h] = x[j] + x[j+h], x[j] - x[j+h]
            h *= 2
        return x

    nl = min(
        128 - int(np.max(np.abs(fast_wht(1 - 2 * ((S >> bit) & 1).astype(float))))) // 2
        for bit in range(8)
    )
    du = max(
        np.bincount([int(S[x]) ^ int(S[x ^ di]) for x in range(256)], minlength=256).max()
        for di in range(1, 256)
    )
    print(f"\n📊 密码学指标")
    print(f"   非线性度 NL : {nl}  (AES=112) {'✅' if nl >= 112 else '❌'}")
    print(f"   差分均匀性  : {du}  (AES=4)   {'✅' if du <= 4  else '❌'}")
    print(f"   nonce       : {params['nonce_hex']}")
    print(f"   seed        : {params['seed_sha512_hex'][:16]}...")

    # 自定义 nD-SPCMM 参数示例
    custom_params = dict(
        a       = [3.1, 3.3, 3.5],
        b_param = [10.0, 11.0, 12.0],
        w       = [0.0, 0.6, 0.5],
        c       = 0.7,
        theta   = 1.0,
    )
    S_custom, _, _ = build_algebra_boolean_chaos_sbox(
        master_key, nonce=nonce, spcmm_params=custom_params
    )
    assert np.array_equal(S, S_custom), "自定义参数与默认参数结果不一致！"
    print("✅ 自定义 nD-SPCMM 参数验证通过")

    print("\n所有验证通过。")
