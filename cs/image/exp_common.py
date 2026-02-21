# exp_common.py
import os, re, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_report(report_path: Path):
    txt = report_path.read_text(encoding="utf-8", errors="ignore")

    def m_float(pat):
        m = re.search(pat, txt)
        return float(m.group(1)) if m else None

    out = {}
    out["MAE"]  = m_float(r"MAE:\s*([0-9.eE+-]+)")
    out["PSNR"] = m_float(r"PSNR:\s*([0-9.eE+-]+)")
    out["SSIM"] = m_float(r"SSIM:\s*([0-9.eE+-]+)")

    out["enc_time_s"] = m_float(r"Encrypt time:\s*([0-9.]+)s")
    out["dec_time_s"] = m_float(r"Decrypt time:\s*([0-9.]+)s")

    out["Entropy"] = m_float(r"Entropy:\s*([0-9.]+)\s*bits")
    out["AdjCorr_H"] = m_float(r"AdjCorr-H:\s*([0-9.eE+-]+)")
    out["AdjCorr_V"] = m_float(r"AdjCorr-V:\s*([0-9.eE+-]+)")
    out["AdjCorr_D"] = m_float(r"AdjCorr-D:\s*([0-9.eE+-]+)")

    out["NPCR"] = m_float(r"NPCR:\s*([0-9.]+)%")
    out["UACI"] = m_float(r"UACI:\s*([0-9.]+)%")

    out["NPCR_key"] = m_float(r"NPCR_key:\s*([0-9.]+)%")
    out["UACI_key"] = m_float(r"UACI_key:\s*([0-9.]+)%")

    out["MAE_wrong"]  = m_float(r"MAE_wrong:\s*([0-9.eE+-]+)")
    out["PSNR_wrong"] = m_float(r"PSNR_wrong:\s*([0-9.eE+-]+)")
    out["SSIM_wrong"] = m_float(r"SSIM_wrong:\s*([0-9.eE+-]+)")
    return out

def write_csv(rows, path: Path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def mean_std(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None, None
    vals = np.array(vals, dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=1)) if vals.size > 1 else 0.0

def plot_xy(xs, ys, out_path: Path, xlabel, ylabel, title):
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
