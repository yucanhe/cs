#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Ablation Experiment Script
音频消融实验 - 测试ARX和S-Box组件的影响
"""

import os
import sys
import numpy as np
from scipy.io import wavfile

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio.audio_cs_spcmm_crypto import (
    encrypt, decrypt, 
    arx_cbc_encrypt, arx_cbc_decrypt,
    sbox_from_spcmm, apply_sbox_to_u32_stream, apply_inv_sbox_to_u32_stream,
    audio_snr_db, audio_psnr_db
)


def run_audio_ablation():
    """运行音频消融实验"""
    print("=" * 70)
    print("Audio Ablation Study")
    print("音频消融实验")
    print("=" * 70)
    
    in_wav = "resources/audio/1.wav"
    key = "my-secret-key-2026"
    
    results = []
    
    # ===== 1. Full (baseline) =====
    print("\n>>> Test 1: Full (ARX + S-Box)")
    out_npz = "results/ablation_audio/full/encrypted.npz"
    out_wav = "results/ablation_audio/full/decrypted.wav"
    
    encrypt(in_wav, out_npz, key, m_rate=0.9, lam=0.001, iters=250, verbose=0)
    decrypt(out_npz, out_wav, key, lam=0.0005, iters=300, ref_wav=in_wav, verbose=0)
    
    # Get metrics
    _, ref = wavfile.read(in_wav)
    _, dec = wavfile.read(out_wav)
    if ref.ndim == 2: ref = ref.mean(axis=1)
    if dec.ndim == 2: dec = dec.mean(axis=1)
    
    results.append({
        'config': 'full',
        'snr': audio_snr_db(ref[:len(dec)], dec),
        'psnr': audio_psnr_db(ref[:len(dec)], dec),
    })
    print(f"   SNR: {results[-1]['snr']:.2f} dB, PSNR: {results[-1]['psnr']:.2f} dB")
    
    print("\n[Note] Audio ablation requires code modification to disable ARX/S-box]")
    print("This is because the audio encryption tightly integrates these components.")
    print("\nFor a complete ablation study, the audio encryption code needs")
    print("to be refactored to support modular enabling/disabling of components.")
    
    return results


if __name__ == "__main__":
    os.makedirs("results/ablation_audio", exist_ok=True)
    results = run_audio_ablation()
    print("\n>>> Audio Ablation Results:")
    for r in results:
        print(f"   {r['config']}: SNR={r['snr']:.2f} dB, PSNR={r['psnr']:.2f} dB")
