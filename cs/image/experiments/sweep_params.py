#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep_params.py
Grid search for demo3.py parameters (cr, lam, iter).

Features:
- Runs demo3.py as subprocess (using current python interpreter)
- Each run outputs to a unique folder under --root
- Parses report.txt and aggregates metrics to CSV
- Produces top-10 lists by PSNR and SSIM

Usage:
  source .venv/bin/activate
  python sweep_params.py --img 1.png --key my-secret-key-2026 --root sweep_runs

You can customize grids via CLI:
  python sweep_params.py --img 1.png --cr_list 0.25,0.5,0.75 --lam_list 0.005,0.01 --iter_list 100,150,200
"""

import argparse
import csv
import os
import re
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple


# -------------------------
# Helpers
# -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def parse_list_floats(s: str) -> List[float]:
    if not s.strip():
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_list_ints(s: str) -> List[int]:
    if not s.strip():
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def run_one(cmd: List[str], cwd: Path) -> Tuple[int, str]:
    """Run command and return (returncode, combined_stdout)."""
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    return p.returncode, p.stdout

def parse_report(report_path: Path) -> Dict[str, Any]:
    """
    Parse demo3.py's report.txt (as printed in your logs).
    Returns dict with keys: MAE, PSNR, SSIM, enc_time, dec_time, entropy, NPCR, UACI, NPCR_key, UACI_key, etc.
    Missing fields become None.
    """
    txt = report_path.read_text(encoding="utf-8", errors="ignore")

    def m_float(pattern: str):
        m = re.search(pattern, txt)
        return float(m.group(1)) if m else None

    def m_str(pattern: str):
        m = re.search(pattern, txt)
        return m.group(1) if m else None

    out = {}
    out["MAE"]  = m_float(r"MAE:\s*([0-9.eE+-]+)")
    out["PSNR"] = m_float(r"PSNR:\s*([0-9.eE+-]+)")
    out["SSIM"] = m_float(r"SSIM:\s*([0-9.eE+-]+)")

    out["MAE_wrong"]  = m_float(r"MAE_wrong:\s*([0-9.eE+-]+)")
    out["PSNR_wrong"] = m_float(r"PSNR_wrong:\s*([0-9.eE+-]+)")
    out["SSIM_wrong"] = m_float(r"SSIM_wrong:\s*([0-9.eE+-]+)")

    out["enc_time_s"] = m_float(r"Encrypt time:\s*([0-9.]+)s")
    out["enc_thr_mib_s"] = m_float(r"Encrypt time:\s*[0-9.]+s\s*throughput:\s*([0-9.]+)\s*MiB/s")

    out["dec_time_s"] = m_float(r"Decrypt time:\s*([0-9.]+)s")
    out["dec_thr_mib_s"] = m_float(r"Decrypt time:\s*[0-9.]+s\s*throughput:\s*([0-9.]+)\s*MiB/s")

    out["Entropy"] = m_float(r"Entropy:\s*([0-9.]+)\s*bits")
    out["AdjCorr_H"] = m_float(r"AdjCorr-H:\s*([0-9.eE+-]+)")
    out["AdjCorr_V"] = m_float(r"AdjCorr-V:\s*([0-9.eE+-]+)")
    out["AdjCorr_D"] = m_float(r"AdjCorr-D:\s*([0-9.eE+-]+)")

    out["NPCR"] = m_float(r"NPCR:\s*([0-9.]+)%")
    out["UACI"] = m_float(r"UACI:\s*([0-9.]+)%")

    out["NPCR_key"] = m_float(r"NPCR_key:\s*([0-9.]+)%")
    out["UACI_key"] = m_float(r"UACI_key:\s*([0-9.]+)%")

    out["u32_mismatch"] = m_float(r"u32 invertibility mismatch count:\s*([0-9]+)")
    out["cu32_img_mismatch"] = m_float(r"Cu32\(image\) mismatch count:\s*([0-9]+)")

    out["cipher_rgba_path"] = m_str(r"cipher_rgba\.png saved:\s*(.+)\s*\(INVERTIBLE FINAL CIPHER IMAGE\)")
    return out

def write_csv(rows: List[Dict[str, Any]], path: Path):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def sort_top(rows: List[Dict[str, Any]], key: str, topn: int = 10, reverse: bool = True):
    def k(r):
        v = r.get(key, None)
        return (-1e18 if v is None else v) if reverse else (1e18 if v is None else v)
    return sorted(rows, key=k, reverse=reverse)[:topn]


# -------------------------
# Main sweep
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=str, default="1.png")
    ap.add_argument("--demo", type=str, default="demo3.py", help="path to demo script")
    ap.add_argument("--root", type=str, default="sweep_runs", help="root folder for all runs")
    ap.add_argument("--key", type=str, default="my-secret-key-2026")
    ap.add_argument("--wrong_key", type=str, default="my-secret-key-2026_wrong")

    # grids (comma-separated)
    ap.add_argument("--cr_list", type=str, default="0.25,0.5,0.75,1.0")
    ap.add_argument("--lam_list", type=str, default="0.005,0.008,0.01,0.012,0.015")
    ap.add_argument("--iter_list", type=str, default="80,120,150,200,250")

    ap.add_argument("--repeat", type=int, default=1, help="repeat each config N times (avg later if you want)")
    ap.add_argument("--timeout", type=int, default=0, help="0=no timeout, else seconds per run")
    ap.add_argument("--stop_on_error", action="store_true")
    args = ap.parse_args()

    cr_list = parse_list_floats(args.cr_list)
    lam_list = parse_list_floats(args.lam_list)
    iter_list = parse_list_ints(args.iter_list)

    if not cr_list or not lam_list or not iter_list:
        print("Empty grid. Please set --cr_list --lam_list --iter_list", file=sys.stderr)
        sys.exit(1)

    cwd = Path(os.getcwd())
    demo_path = (cwd / args.demo).resolve()
    if not demo_path.exists():
        print(f"demo script not found: {demo_path}", file=sys.stderr)
        sys.exit(1)

    root = (cwd / args.root).resolve()
    ensure_dir(root)

    results: List[Dict[str, Any]] = []
    total = len(cr_list) * len(lam_list) * len(iter_list) * max(1, args.repeat)
    run_id = 0

    print(f"[SWEEP] demo: {demo_path.name}")
    print(f"[SWEEP] root: {root}")
    print(f"[SWEEP] total runs: {total}\n")

    for cr in cr_list:
        for lam in lam_list:
            for it in iter_list:
                for rep in range(args.repeat):
                    run_id += 1
                    tag = f"run_{run_id:04d}_cr{cr:.3f}_lam{lam:.4f}_iter{it}_rep{rep+1}"
                    out_dir = root / tag
                    ensure_dir(out_dir)

                    cmd = [
                        sys.executable,  # IMPORTANT: use current interpreter (venv-safe)
                        str(demo_path),
                        "--img", args.img,
                        "--out", str(out_dir),
                        "--key", args.key,
                        "--wrong_key", args.wrong_key,
                        "--cr", str(cr),
                        "--lam", str(lam),
                        "--iter", str(it),
                    ]

                    print(f"[{run_id}/{total}] {tag}")
                    t0 = time.perf_counter()

                    try:
                        if args.timeout and args.timeout > 0:
                            p = subprocess.run(
                                cmd,
                                cwd=str(cwd),
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True,
                                timeout=args.timeout,
                            )
                            rc, out = p.returncode, p.stdout
                        else:
                            rc, out = run_one(cmd, cwd=cwd)
                    except subprocess.TimeoutExpired as e:
                        rc = 124
                        out = (e.stdout or "") + "\n[TIMEOUT]\n"
                    except Exception as e:
                        rc = 125
                        out = f"[EXCEPTION] {e}\n"

                    dt = time.perf_counter() - t0

                    # Save console log
                    (out_dir / "console.log").write_text(out, encoding="utf-8", errors="ignore")

                    row: Dict[str, Any] = {
                        "run_id": run_id,
                        "tag": tag,
                        "img": args.img,
                        "cr": cr,
                        "lam": lam,
                        "iter": it,
                        "rep": rep + 1,
                        "return_code": rc,
                        "wall_time_s": dt,
                        "out_dir": str(out_dir),
                    }

                    report_path = out_dir / "report.txt"
                    if rc == 0 and report_path.exists():
                        metrics = parse_report(report_path)
                        row.update(metrics)
                        print(f"   -> PSNR={row.get('PSNR')}  SSIM={row.get('SSIM')}  enc={row.get('enc_time_s')}s  dec={row.get('dec_time_s')}s")
                    else:
                        print(f"   -> FAILED rc={rc}  (see {out_dir/'console.log'})")

                    results.append(row)

                    if rc != 0 and args.stop_on_error:
                        print("[SWEEP] stop_on_error enabled, aborting.", file=sys.stderr)
                        write_csv(results, root / "results_partial.csv")
                        sys.exit(rc)

    # Write full CSV
    results_csv = root / "results.csv"
    write_csv(results, results_csv)

    # Top-10
    ok_rows = [r for r in results if r.get("return_code") == 0 and r.get("PSNR") is not None]
    top_psnr = sort_top(ok_rows, "PSNR", topn=10, reverse=True)
    top_ssim = sort_top(ok_rows, "SSIM", topn=10, reverse=True)

    write_csv(top_psnr, root / "top10_by_psnr.csv")
    write_csv(top_ssim, root / "top10_by_ssim.csv")

    print("\n[SWEEP] done.")
    print(f" - results: {results_csv}")
    print(f" - top10_by_psnr: {root/'top10_by_psnr.csv'}")
    print(f" - top10_by_ssim: {root/'top10_by_ssim.csv'}")

    if top_psnr:
        best = top_psnr[0]
        print("\n[BEST by PSNR]")
        print(f" tag: {best['tag']}")
        print(f" cr={best['cr']}  lam={best['lam']}  iter={best['iter']}")
        print(f" PSNR={best.get('PSNR')}  SSIM={best.get('SSIM')}  MAE={best.get('MAE')}")
        print(f" out_dir: {best['out_dir']}")


if __name__ == "__main__":
    main()
