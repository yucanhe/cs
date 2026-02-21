#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Ablation Experiment
音频消融实验 - 比较不同配置
"""

import os
import sys
import time
import json
import numpy as np
from scipy.io import wavfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio.audio_cs_spcmm_crypto import encrypt, decrypt, audio_snr_db, audio_psnr_db


def compute_audio_correlation(wav_path):
    """计算相邻样本相关性"""
    fs, audio = wavfile.read(wav_path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    
    audio = audio.astype(np.float64)
    
    # 计算相邻相关性
    n = len(audio) - 1
    corr = np.corrcoef(audio[:-1], audio[1:])[0, 1]
    
    return corr


def run_audio_ablation():
    """运行音频消融实验"""
    print("=" * 70)
    print("Audio Ablation Study")
    print("音频消融实验")
    print("=" * 70)
    
    audio_path = "resources/audio/1.wav"
    key = "my-secret-key-2026"
    out_dir = "results/audio_ablation"
    os.makedirs(out_dir, exist_ok=True)
    
    results = []
    
    # 1. Full encryption (baseline)
    print("\n>>> [1/4] Full Encryption (Baseline)")
    npz_path = os.path.join(out_dir, "ablation_full.npz")
    encrypt(audio_path, npz_path, key, m_rate=0.9, iters=250, verbose=0)
    
    # Decrypt
    dec_path = os.path.join(out_dir, "ablation_full_dec.wav")
    decrypt(npz_path, dec_path, key, lam=0.0005, iters=300, ref_wav=audio_path, verbose=0)
    
    # Metrics
    fs_orig, orig = wavfile.read(audio_path)
    fs_dec, dec = wavfile.read(dec_path)
    
    snr = audio_snr_db(orig, dec)
    psnr = audio_psnr_db(orig, dec)
    corr_orig = compute_audio_correlation(audio_path)
    corr_enc = compute_audio_correlation(npz_path.replace('.npz', '_enc.wav'))
    
    results.append({
        'config': 'full',
        'snr': snr,
        'psnr': psnr,
        'corr_plain': corr_orig,
    })
    
    print(f"   SNR: {snr:.2f} dB, PSNR: {psnr:.2f} dB")
    
    # 2. High compression (m_rate=0.5)
    print("\n>>> [2/4] High Compression (m_rate=0.5)")
    npz_path2 = os.path.join(out_dir, "ablation_cr05.npz")
    encrypt(audio_path, npz_path2, key, m_rate=0.5, iters=250, verbose=0)
    
    dec_path2 = os.path.join(out_dir, "ablation_cr05_dec.wav")
    decrypt(npz_path2, dec_path2, key, lam=0.0005, iters=300, ref_wav=audio_path, verbose=0)
    
    fs_orig2, orig2 = wavfile.read(audio_path)
    fs_dec2, dec2 = wavfile.read(dec_path2)
    
    snr2 = audio_snr_db(orig2, dec2)
    psnr2 = audio_psnr_db(orig2, dec2)
    
    results.append({
        'config': 'cr_0.5',
        'snr': snr2,
        'psnr': psnr2,
        'corr_plain': corr_orig,
    })
    
    print(f"   SNR: {snr2:.2f} dB, PSNR: {psnr2:.2f} dB")
    
    # 3. Low compression (m_rate=0.99)
    print("\n>>> [3/4] Low Compression (m_rate=0.99)")
    npz_path3 = os.path.join(out_dir, "ablation_cr099.npz")
    encrypt(audio_path, npz_path3, key, m_rate=0.99, iters=250, verbose=0)
    
    dec_path3 = os.path.join(out_dir, "ablation_cr099_dec.wav")
    decrypt(npz_path3, dec_path3, key, lam=0.0005, iters=300, ref_wav=audio_path, verbose=0)
    
    fs_orig3, orig3 = wavfile.read(audio_path)
    fs_dec3, dec3 = wavfile.read(dec_path3)
    
    snr3 = audio_snr_db(orig3, dec3)
    psnr3 = audio_psnr_db(orig3, dec3)
    
    results.append({
        'config': 'cr_0.99',
        'snr': snr3,
        'psnr': psnr3,
        'corr_plain': corr_orig,
    })
    
    print(f"   SNR: {snr3:.2f} dB, PSNR: {psnr3:.2f} dB")
    
    # 4. No compression (just encryption)
    print("\n>>> [4/4] No Compression (m_rate=1.0)")
    npz_path4 = os.path.join(out_dir, "ablation_nocomp.npz")
    encrypt(audio_path, npz_path4, key, m_rate=1.0, iters=250, verbose=0)
    
    dec_path4 = os.path.join(out_dir, "ablation_nocomp_dec.wav")
    decrypt(npz_path4, dec_path4, key, lam=0.0005, iters=300, ref_wav=audio_path, verbose=0)
    
    fs_orig4, orig4 = wavfile.read(audio_path)
    fs_dec4, dec4 = wavfile.read(dec_path4)
    
    snr4 = audio_snr_db(orig4, dec4)
    psnr4 = audio_psnr_db(orig4, dec4)
    
    results.append({
        'config': 'no_compression',
        'snr': snr4,
        'psnr': psnr4,
        'corr_plain': corr_orig,
    })
    
    print(f"   SNR: {snr4:.2f} dB, PSNR: {psnr4:.2f} dB")
    
    # Save results
    with open(os.path.join(out_dir, "ablation_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Audio Ablation Summary")
    print("=" * 70)
    
    print(f"\n{'Config':<20} {'SNR (dB)':<12} {'PSNR (dB)':<12}")
    print("-" * 50)
    
    for r in results:
        print(f"{r['config']:<20} {r['snr']:<12.2f} {r['psnr']:<12.2f}")
    
    print("\n" + "=" * 70)
    
    return results


if __name__ == "__main__":
    run_audio_ablation()
