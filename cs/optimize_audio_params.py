#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频加密参数优化实验
"""

import os
import sys
import json
import time
import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio.audio_cs_spcmm_crypto import encrypt, decrypt, audio_snr_db, audio_psnr_db


def run_audio_optimize():
    """运行音频参数优化"""
    print("=" * 70)
    print("Audio Parameter Optimization")
    print("音频加密参数优化")
    print("=" * 70)
    
    audio_path = "resources/audio/1.wav"
    key = "my-secret-key-2026"
    out_dir = "results/optimize_audio"
    os.makedirs(out_dir, exist_ok=True)
    
    results = []
    
    # 1. m_rate 优化
    print("\n>>> [1/3] 优化 m_rate (压缩率)")
    
    m_rates = [0.7, 0.8, 0.9, 0.95, 0.99]
    
    for m_rate in m_rates:
        print(f"\n   Testing m_rate={m_rate}")
        
        npz_path = os.path.join(out_dir, f"audio_m{m_rate}.npz")
        
        # 加密
        t0 = time.time()
        encrypt(audio_path, npz_path, key, m_rate=m_rate, iters=250, verbose=0)
        enc_time = time.time() - t0
        
        # 解密
        dec_path = os.path.join(out_dir, f"audio_m{m_rate}_dec.wav")
        t1 = time.time()
        decrypt(npz_path, dec_path, key, lam=0.0005, iters=300, ref_wav=audio_path, verbose=0)
        dec_time = time.time() - t1
        
        # 计算质量
        try:
            _, orig = wavfile.read(audio_path)
            _, dec = wavfile.read(dec_path)
            
            if orig.ndim == 2:
                orig = orig.mean(axis=1)
            if dec.ndim == 2:
                dec = dec.mean(axis=1)
            
            # 确保长度一致
            min_len = min(len(orig), len(dec))
            orig = orig[:min_len]
            dec = dec[:min_len]
            
            snr = audio_snr_db(orig.astype(np.float64), dec.astype(np.float64))
            psnr = audio_psnr_db(orig.astype(np.float64), dec.astype(np.float64))
            
            print(f"   SNR: {snr:.2f} dB, PSNR: {psnr:.2f} dB, Time: {enc_time:.1f}s/{dec_time:.1f}s")
            
            results.append({
                'param': 'm_rate',
                'value': m_rate,
                'snr': snr,
                'psnr': psnr,
                'enc_time': enc_time,
                'dec_time': dec_time,
            })
        except Exception as e:
            print(f"   Error: {e}")
    
    # 2. iters 优化
    print("\n>>> [2/3] 优化 iters (迭代次数)")
    
    iters_list = [100, 150, 200, 250, 300]
    
    for iters in iters_list:
        print(f"\n   Testing iters={iters}")
        
        npz_path = os.path.join(out_dir, f"audio_i{iters}.npz")
        
        t0 = time.time()
        encrypt(audio_path, npz_path, key, m_rate=0.9, iters=iters, verbose=0)
        enc_time = time.time() - t0
        
        dec_path = os.path.join(out_dir, f"audio_i{iters}_dec.wav")
        t1 = time.time()
        decrypt(npz_path, dec_path, key, lam=0.0005, iters=300, ref_wav=audio_path, verbose=0)
        dec_time = time.time() - t1
        
        try:
            _, orig = wavfile.read(audio_path)
            _, dec = wavfile.read(dec_path)
            
            if orig.ndim == 2:
                orig = orig.mean(axis=1)
            if dec.ndim == 2:
                dec = dec.mean(axis=1)
            
            min_len = min(len(orig), len(dec))
            orig = orig[:min_len]
            dec = dec[:min_len]
            
            snr = audio_snr_db(orig.astype(np.float64), dec.astype(np.float64))
            psnr = audio_psnr_db(orig.astype(np.float64), dec.astype(np.float64))
            
            print(f"   SNR: {snr:.2f} dB, PSNR: {psnr:.2f} dB, Time: {enc_time:.1f}s/{dec_time:.1f}s")
            
            results.append({
                'param': 'iters',
                'value': iters,
                'snr': snr,
                'psnr': psnr,
                'enc_time': enc_time,
                'dec_time': dec_time,
            })
        except Exception as e:
            print(f"   Error: {e}")
    
    # 3. nperseg 优化
    print("\n>>> [3/3] 优化 nperseg (STFT窗口)")
    
    nperseg_list = [512, 1024, 2048]
    
    for nperseg in nperseg_list:
        print(f"\n   Testing nperseg={nperseg}")
        
        npz_path = os.path.join(out_dir, f"audio_n{_nperseg}.npz")
        
        t0 = time.time()
        encrypt(audio_path, npz_path, key, m_rate=0.9, iters=250, nperseg=nperseg, verbose=0)
        enc_time = time.time() - t0
        
        dec_path = os.path.join(out_dir, f"audio_n{nperseg}_dec.wav")
        t1 = time.time()
        decrypt(npz_path, dec_path, key, lam=0.0005, iters=300, ref_wav=audio_path, verbose=0)
        dec_time = time.time() - t1
        
        try:
            _, orig = wavfile.read(audio_path)
            _, dec = wavfile.read(dec_path)
            
            if orig.ndim == 2:
                orig = orig.mean(axis=1)
            if dec.ndim == 2:
                dec = dec.mean(axis=1)
            
            min_len = min(len(orig), len(dec))
            orig = orig[:min_len]
            dec = dec[:min_len]
            
            snr = audio_snr_db(orig.astype(np.float64), dec.astype(np.float64))
            psnr = audio_psnr_db(orig.astype(np.float64), dec.astype(np.float64))
            
            print(f"   SNR: {snr:.2f} dB, PSNR: {psnr:.2f} dB, Time: {enc_time:.1f}s/{dec_time:.1f}s")
            
            results.append({
                'param': 'nperseg',
                'value': nperseg,
                'snr': snr,
                'psnr': psnr,
                'enc_time': enc_time,
                'dec_time': dec_time,
            })
        except Exception as e:
            print(f"   Error: {e}")
    
    # 保存结果
    with open(os.path.join(out_dir, "audio_optimization.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印摘要
    print("\n" + "=" * 70)
    print("Audio Optimization Summary")
    print("=" * 70)
    
    # 按m_rate分组打印
    print("\n>>> m_rate 影响:")
    m_rate_results = [r for r in results if r['param'] == 'm_rate']
    for r in sorted(m_rate_results, key=lambda x: x['value']):
        print(f"   m_rate={r['value']}: SNR={r['snr']:.2f}dB, PSNR={r['psnr']:.2f}dB")
    
    print("\n>>> iters 影响:")
    iters_results = [r for r in results if r['param'] == 'iters']
    for r in sorted(iters_results, key=lambda x: x['value']):
        print(f"   iters={r['value']}: SNR={r['snr']:.2f}dB, PSNR={r['psnr']:.2f}dB")
    
    return results


if __name__ == "__main__":
    run_audio_optimize()
