#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import time
import argparse
import hashlib
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft
from scipy.fftpack import dct, idct


# ============================================================
# 0) Hash / seed utilities (Py3.8 safe)
# ============================================================

def sha256_bytes(b):
    h = hashlib.sha256()
    h.update(b)
    return h.digest()

def derive_seed_bytes(key_str, tag_str):
    return sha256_bytes((key_str + "|" + tag_str).encode("utf-8"))

def seed_to_u64(seed_bytes):
    return int.from_bytes(seed_bytes[:8], byteorder="little", signed=False)

def u64_to_unit(x):
    return ((x % (2**53 - 1)) / float(2**53))

def normalize_audio_int16_to_float(x):
    x = x.astype(np.float64)
    mx = np.max(np.abs(x)) + 1e-12
    return x / mx

def float_to_int16(x):
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


# ============================================================
# 0b) Quality metrics
# ============================================================

def audio_snr_db(x_ref_i16, x_hat_i16):
    x = x_ref_i16.astype(np.float64)
    xh = x_hat_i16.astype(np.float64)
    n = min(len(x), len(xh))
    x = x[:n]; xh = xh[:n]
    num = np.sum(x * x) + 1e-12
    den = np.sum((x - xh) * (x - xh)) + 1e-12
    return 10.0 * np.log10(num / den)

def audio_psnr_db(x_ref_i16, x_hat_i16):
    x = x_ref_i16.astype(np.float64)
    xh = x_hat_i16.astype(np.float64)
    n = min(len(x), len(xh))
    x = x[:n]; xh = xh[:n]
    peak = np.max(np.abs(x)) + 1e-12
    mse = np.mean((x - xh) * (x - xh)) + 1e-12
    return 20.0 * np.log10(peak) - 10.0 * np.log10(mse)

def audio_mse(x_ref_i16, x_hat_i16):
    x = x_ref_i16.astype(np.float64)
    xh = x_hat_i16.astype(np.float64)
    n = min(len(x), len(xh))
    x = x[:n]; xh = xh[:n]
    return float(np.mean((x - xh) * (x - xh)))


# ============================================================
# 1) 3D-SPCMM (streaming iterator)
# ============================================================

class SPCMM3D(object):
    def __init__(self, x0, y0, z0,
                 a=(3.1, 3.3, 3.5),
                 b=(10.0, 11.0, 12.0),
                 w=(0.6, 0.5, 0.6),
                 c=0.7):
        self.x = float(x0) % 1.0
        self.y = float(y0) % 1.0
        self.z = float(z0) % 1.0
        self.a1, self.a2, self.a3 = map(float, a)
        self.b1, self.b2, self.b3 = map(float, b)
        self.w1, self.w2, self.w3 = map(float, w)
        self.c = float(c)

    def step(self):
        z = (self.a3 * math.sin(self.b3 * self.z) + self.c) % 1.0
        y = (self.a2 * math.sin(self.b2 * self.y) + self.w3 * (z * z) + self.c) % 1.0
        x = (self.a1 * math.sin(self.b1 * self.x) + self.w1 * (y * y) + self.w2 * (z * z) + self.c) % 1.0
        self.x, self.y, self.z = x, y, z
        return x, y, z


def spcmm_from_key(key_str, tag_str):
    seed = derive_seed_bytes(key_str, tag_str)
    u = seed_to_u64(seed)
    u1 = (u ^ 0xA5A5A5A5A5A5A5A5) & 0xFFFFFFFFFFFFFFFF
    u2 = ((u << 1) ^ 0x0123456789ABCDEF) & 0xFFFFFFFFFFFFFFFF
    u3 = ((u >> 1) ^ 0xFEDCBA9876543210) & 0xFFFFFFFFFFFFFFFF
    x0 = u64_to_unit(u1)
    y0 = u64_to_unit(u2)
    z0 = u64_to_unit(u3)
    x0 = 0.123456789 if x0 <= 0.0 else (0.987654321 if x0 >= 1.0 else x0)
    y0 = 0.456789123 if y0 <= 0.0 else (0.876543219 if y0 >= 1.0 else y0)
    z0 = 0.789123456 if z0 <= 0.0 else (0.765432198 if z0 >= 1.0 else z0)
    return SPCMM3D(x0, y0, z0)


def spcmm_float_seq(key_str, tag_str, length, discard=256, take_dim=0):
    gen = spcmm_from_key(key_str, tag_str)
    for _ in range(int(discard)):
        gen.step()
    out = np.empty(int(length), dtype=np.float64)
    for i in range(int(length)):
        x, y, z = gen.step()
        if take_dim == 0:
            out[i] = x
        elif take_dim == 1:
            out[i] = y
        elif take_dim == 2:
            out[i] = z
        else:
            out[i] = (x + y + z) / 3.0
    return out


def spcmm_u32_keystream(key_str, tag_str, count, discard=256):
    gen = spcmm_from_key(key_str, tag_str)
    for _ in range(int(discard)):
        gen.step()
    ks = np.empty(int(count), dtype=np.uint32)
    for i in range(int(count)):
        x, y, z = gen.step()
        vx = int((x * (2**32 - 1))) & 0xFFFFFFFF
        vy = int((y * (2**32 - 1))) & 0xFFFFFFFF
        vz = int((z * (2**32 - 1))) & 0xFFFFFFFF
        v = (vx ^ ((vy << 13) & 0xFFFFFFFF) ^ ((vz >> 7) & 0xFFFFFFFF)) & 0xFFFFFFFF
        ks[i] = np.uint32(v)
    return ks


# ============================================================
# 2) Measurement matrix Phi from chaos
# ============================================================

def make_measurement_matrix_from_spcmm(key_str, tag_str, m, n):
    chaos = spcmm_float_seq(key_str, tag_str, m * n, discard=256, take_dim=0)
    A = chaos.reshape(m, n)
    A = (A - 0.5) / 0.5
    col = np.linalg.norm(A, axis=0, keepdims=True) + 1e-12
    return A / col


# ============================================================
# 3) Phase permutation
# ============================================================

def perm_indices_from_spcmm(key_str, tag_str, L):
    chaos = spcmm_float_seq(key_str, tag_str, L, discard=128, take_dim=1)
    return np.argsort(chaos)

def inv_perm(p):
    inv = np.zeros_like(p)
    inv[p] = np.arange(len(p))
    return inv


# ============================================================
# 4) FISTA
# ============================================================

def soft_threshold(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)

def fista(A, y, lam=0.001, max_iter=250):
    m, n = A.shape
    x = np.zeros(n, dtype=np.float64)
    z = x.copy()
    t = 1.0
    AtA = A.T @ A
    L = np.linalg.norm(AtA, 2)
    step = 1.0 / (L + 1e-12)
    for _ in range(int(max_iter)):
        grad = A.T @ (A @ z - y)
        x_new = soft_threshold(z - step * grad, lam * step)
        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
    return x


# ============================================================
# 5) ARX-CBC (uint32) diffusion (no underflow warning)
# ============================================================

U32_MASK = np.uint64(0xFFFFFFFF)
U32_MOD  = np.uint64(1) << np.uint64(32)

def rotl32(x, r):
    r &= 31
    x = np.uint32(x)
    return np.uint32(((x << np.uint32(r)) & np.uint32(0xFFFFFFFF)) | (x >> np.uint32(32 - r)))

def rotr32(x, r):
    r &= 31
    x = np.uint32(x)
    return np.uint32((x >> np.uint32(r)) | ((x << np.uint32(32 - r)) & np.uint32(0xFFFFFFFF)))

def add_u32(a, b):
    return np.uint32((np.uint64(a) + np.uint64(b)) & U32_MASK)

def sub_u32(a, b):
    return np.uint32((np.uint64(a) + U32_MOD - np.uint64(b)) & U32_MASK)

def arx_cbc_encrypt(u32_plain, key_str, tag_str):
    x = np.asarray(u32_plain, dtype=np.uint32).copy()
    n = x.size
    ks = spcmm_u32_keystream(key_str, tag_str + "|arx", n + 1, discard=512)
    iv = ks[0]
    prev = np.uint32(iv)
    out = np.empty(n, dtype=np.uint32)
    for i in range(n):
        k = ks[i + 1]
        r = int(k & np.uint32(31))
        t = add_u32(add_u32(x[i], k), prev)
        t = rotl32(t, r)
        c = np.uint32(t ^ rotl32(k, (r + 7) & 31))
        out[i] = c
        prev = c
    return out, np.uint32(iv)

def arx_cbc_decrypt(u32_cipher, key_str, tag_str, iv):
    c = np.asarray(u32_cipher, dtype=np.uint32)
    n = c.size
    ks = spcmm_u32_keystream(key_str, tag_str + "|arx", n + 1, discard=512)
    prev = np.uint32(iv)
    out = np.empty(n, dtype=np.uint32)
    for i in range(n):
        k = ks[i + 1]
        r = int(k & np.uint32(31))
        t = np.uint32(c[i] ^ rotl32(k, (r + 7) & 31))
        t = rotr32(t, r)
        u = sub_u32(sub_u32(t, k), prev)
        out[i] = u
        prev = c[i]
    return out


# ============================================================
# 6) Dynamic S-Box
# ============================================================

def sbox_from_spcmm(key_str, tag_str):
    chaos = spcmm_float_seq(key_str, tag_str + "|sbox", 256, discard=1024, take_dim=2)
    order = np.argsort(chaos)
    sbox = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        sbox[i] = np.uint8(order[i])
    inv = np.zeros(256, dtype=np.uint8)
    inv[sbox] = np.arange(256, dtype=np.uint8)
    return sbox, inv

def apply_sbox_to_u32_stream(u32_arr, sbox):
    b = np.asarray(u32_arr, dtype=np.uint32).view(np.uint8).reshape(-1)
    return np.asarray(sbox[b], dtype=np.uint8).view(np.uint32)

def apply_inv_sbox_to_u32_stream(u32_arr, inv_sbox):
    b = np.asarray(u32_arr, dtype=np.uint32).view(np.uint8).reshape(-1)
    return np.asarray(inv_sbox[b], dtype=np.uint8).view(np.uint32)


# ============================================================
# 7) Encrypt / Decrypt
# ============================================================

def encrypt(in_wav, out_npz, key,
            m_rate=0.90, lam=0.001, iters=250,
            nperseg=1024, noverlap=512,
            f_max=512, phi_group=8,
            save_float32=1, verbose=1):

    fs, x = wavfile.read(in_wav)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype(np.int16)
    orig_len = int(len(x))
    x_f = normalize_audio_int16_to_float(x)

    # STFT (invertible-friendly)
    f, tt, Z = stft(
        x_f, fs=fs,
        window="hann",
        nperseg=nperseg, noverlap=noverlap,
        boundary="zeros",
        padded=True
    )
    F, T = Z.shape
    f_use = min(int(f_max), F)

    phase = np.angle(Z[:f_use, :])
    mag = np.abs(Z[:f_use, :])

    # phase perm
    phase_enc = phase.copy()
    for fi in range(f_use):
        p = perm_indices_from_spcmm(key, "phase|f=%d|T=%d" % (fi, T), T)
        phase_enc[fi, :] = phase_enc[fi, p]

    # Time-DCT + CS
    n = T
    m = max(1, int(round(float(m_rate) * n)))
    groups = int(math.ceil(f_use / float(phi_group)))
    Y = np.zeros((f_use, m), dtype=np.float64)

    t0 = time.time()
    for g in range(groups):
        f0 = g * phi_group
        f1 = min(f_use, (g + 1) * phi_group)
        if verbose and (g % 10 == 0):
            print("  [encrypt] group %d/%d  freq [%d,%d)" % (g, groups, f0, f1))

        Phi = make_measurement_matrix_from_spcmm(key, "phi|g=%d|m=%d|n=%d" % (g, m, n), m, n)

        for fi in range(f0, f1):
            s = dct(mag[fi, :], type=2, norm="ortho").astype(np.float64)
            y = Phi @ s
            Y[fi, :] = y

    t1 = time.time()

    # quantize Y -> uint32
    y_min = float(np.min(Y))
    y_ptp = float(np.ptp(Y)) + 1e-12
    Y_norm = (Y - y_min) / y_ptp
    Q = np.clip(np.round(Y_norm * (2**32 - 1)), 0, 2**32 - 1).astype(np.uint32)

    # ARX-CBC
    q_flat = Q.reshape(-1)
    arx_out, iv = arx_cbc_encrypt(q_flat, key, "meas")
    arx_out = arx_out.reshape(Q.shape)

    # S-Box
    sbox, _inv = sbox_from_spcmm(key, "dyn")
    C = apply_sbox_to_u32_stream(arx_out.reshape(-1), sbox).reshape(Q.shape)

    meta = dict(
        fs=int(fs),
        orig_len=int(orig_len),
        nperseg=int(nperseg),
        noverlap=int(noverlap),
        F=int(F),
        T=int(T),
        f_use=int(f_use),
        m=int(m),
        m_rate=float(m_rate),
        lam=float(lam),
        iters=int(iters),
        f_max=int(f_max),
        phi_group=int(phi_group),
        y_min=y_min,
        y_ptp=y_ptp,
        iv=int(iv),
        pipeline="STFT(hann,zeros,padded) + phase-perm + TimeDCT-CS + quant(u32) + ARX-CBC + SBox",
    )

    if int(save_float32) == 1:
        np.savez(out_npz,
                 C=C.astype(np.uint32),
                 phase_enc=phase_enc.astype(np.float32),
                 meta=json.dumps(meta))
    else:
        np.savez(out_npz,
                 C=C.astype(np.uint32),
                 phase_enc=phase_enc.astype(np.float64),
                 meta=json.dumps(meta))

    if verbose:
        b = C.astype(np.uint32).view(np.uint8).reshape(-1)
        hist = np.bincount(b, minlength=256).astype(np.float64)
        p = hist / (hist.sum() + 1e-12)
        p = p[p > 0]
        H = float(-np.sum(p * np.log2(p)))
        print("\n=== Encrypt Summary ===")
        print("Saved: %s Encrypt time: %.3fs" % (out_npz, (t1 - t0)))
        print("Entropy(8-bit view): %.4f" % H)


def decrypt(in_npz, out_wav, key,
            lam=0.0005, iters=300,
            ref_wav=None,
            verbose=1):

    data = np.load(in_npz, allow_pickle=True)
    C = data["C"].astype(np.uint32)
    phase_enc = data["phase_enc"].astype(np.float64)
    meta = json.loads(str(data["meta"]))

    fs = int(meta["fs"])
    orig_len = int(meta.get("orig_len", 0))
    nperseg = int(meta["nperseg"])
    noverlap = int(meta["noverlap"])
    T = int(meta["T"])
    f_use = int(meta["f_use"])
    m = int(meta["m"])
    f_max = int(meta["f_max"])
    phi_group = int(meta["phi_group"])
    y_min = float(meta["y_min"])
    y_ptp = float(meta["y_ptp"])
    iv = np.uint32(int(meta["iv"]))

    # inverse S-Box
    _sbox, inv_sbox = sbox_from_spcmm(key, "dyn")
    arx_u32 = apply_inv_sbox_to_u32_stream(C.reshape(-1), inv_sbox).reshape(C.shape)

    # ARX-CBC decrypt
    q_flat = arx_cbc_decrypt(arx_u32.reshape(-1), key, "meas", iv).reshape(arx_u32.shape)

    # dequantize
    Y_norm = q_flat.astype(np.float64) / float(2**32 - 1)
    Y = Y_norm * y_ptp + y_min

    # inverse phase perm
    phase = phase_enc.copy()
    for fi in range(f_use):
        p = perm_indices_from_spcmm(key, "phase|f=%d|T=%d" % (fi, T), T)
        ip = inv_perm(p)
        phase[fi, :] = phase[fi, ip]

    # reconstruct mag
    n = T
    groups = int(math.ceil(f_use / float(phi_group)))
    mag_hat = np.zeros((f_use, T), dtype=np.float64)

    t0 = time.time()
    for g in range(groups):
        f0 = g * phi_group
        f1 = min(f_use, (g + 1) * phi_group)
        if verbose and (g % 10 == 0):
            print("  [decrypt] group %d/%d  freq [%d,%d)" % (g, groups, f0, f1))

        Phi = make_measurement_matrix_from_spcmm(key, "phi|g=%d|m=%d|n=%d" % (g, m, n), m, n)

        for fi in range(f0, f1):
            y = Y[fi, :]
            # y 的幅度大 -> lam 稍大；幅度小 -> lam 更小，避免被“压死”
            lam_i = float(lam) * (np.linalg.norm(y) / (np.sqrt(len(y)) + 1e-12))
            lam_i = max(lam_i, float(lam) * 0.2)  # 下限
            lam_i = min(lam_i, float(lam) * 3.0)  # 上限
            s_hat = fista(Phi, y, lam=lam_i, max_iter=iters)
            mag_hat[fi, :] = idct(s_hat, type=2, norm="ortho")

    # ✅ 关键修复：幅度物理约束（减少金属声/滋滋声）
    mag_hat = np.maximum(mag_hat, 0.0)

    t1 = time.time()

    # rebuild complex STFT
    Z_hat = mag_hat * (np.cos(phase[:f_use, :]) + 1j * np.sin(phase[:f_use, :]))
    F_full = nperseg // 2 + 1
    Z_full = np.zeros((F_full, T), dtype=np.complex128)
    Z_full[:f_use, :] = Z_hat

    # ISTFT
    _, x_hat = istft(
        Z_full, fs=fs,
        window="hann",
        nperseg=nperseg, noverlap=noverlap
    )

    x_hat = x_hat.astype(np.float64)
    if orig_len > 0:
        x_hat = x_hat[:orig_len]

    x_hat = x_hat / (np.max(np.abs(x_hat)) + 1e-12)
    x_hat_i16 = float_to_int16(x_hat)
    wavfile.write(out_wav, fs, x_hat_i16)

    if verbose:
        print("\n=== Decrypt Summary ===")
        print("Saved: %s" % out_wav)
        print("Decrypt time: %.3fs" % (t1 - t0))

    # ✅ ref quality
    if ref_wav is not None:
        fs0, xr = wavfile.read(ref_wav)
        if xr.ndim == 2:
            xr = xr.mean(axis=1)
        xr = xr.astype(np.int16)

        print("\n=== Quality vs Reference ===")
        print("SNR : %.3f dB" % audio_snr_db(xr, x_hat_i16))
        print("PSNR: %.3f dB" % audio_psnr_db(xr, x_hat_i16))
        print("MSE : %.3f" % audio_mse(xr, x_hat_i16))


# ============================================================
# 9) CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_e = sub.add_parser("encrypt")
    ap_e.add_argument("--in_wav", required=True)
    ap_e.add_argument("--out_npz", required=True)
    ap_e.add_argument("--key", required=True)
    ap_e.add_argument("--m_rate", type=float, default=0.90)
    ap_e.add_argument("--lam", type=float, default=0.001)
    ap_e.add_argument("--iter", type=int, default=250)
    ap_e.add_argument("--nperseg", type=int, default=1024)
    ap_e.add_argument("--noverlap", type=int, default=512)
    ap_e.add_argument("--f_max", type=int, default=512)
    ap_e.add_argument("--phi_group", type=int, default=8)
    ap_e.add_argument("--save_float32", type=int, default=1)

    ap_d = sub.add_parser("decrypt")
    ap_d.add_argument("--in_npz", required=True)
    ap_d.add_argument("--out_wav", required=True)
    ap_d.add_argument("--key", required=True)
    ap_d.add_argument("--lam", type=float, default=0.0005)
    ap_d.add_argument("--iter", type=int, default=300)
    ap_d.add_argument("--ref_wav", default=None)

    args = ap.parse_args()

    if args.cmd == "encrypt":
        print("\n=== Encrypt (STFT + Time-CS + ARX-CBC + S-Box) ===")
        encrypt(args.in_wav, args.out_npz, args.key,
                m_rate=args.m_rate, lam=args.lam, iters=args.iter,
                nperseg=args.nperseg, noverlap=args.noverlap,
                f_max=args.f_max, phi_group=args.phi_group,
                save_float32=args.save_float32, verbose=1)
    else:
        print("\n=== Decrypt (STFT + Time-CS + ARX-CBC + S-Box) ===")
        decrypt(args.in_npz, args.out_wav, args.key,
                lam=args.lam, iters=args.iter,
                ref_wav=args.ref_wav,
                verbose=1)

if __name__ == "__main__":
    main()
