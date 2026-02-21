# exp_3_repeat.py
import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

# 确保可以从父目录导入 exp_common（脚本已移动到 image/experiments/）
BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from exp_common import ensure_dir, parse_report, write_csv, mean_std

def run_demo(demo, img, out_dir, key, wrong_key, cr, lam, it, extra_args):
    cmd = [sys.executable, demo, "--img", img, "--out", str(out_dir),
           "--key", key, "--wrong_key", wrong_key, "--cr", str(cr), "--lam", str(lam), "--iter", str(it)]
    cmd += extra_args
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (out_dir / "console.log").write_text(p.stdout, encoding="utf-8", errors="ignore")
    return p.returncode

def main():
    ap = argparse.ArgumentParser()
    demo_default = str((Path(__file__).resolve().parent.parent / "demo" / "demo3.py"))
    ap.add_argument("--demo", default=demo_default)
    ap.add_argument("--img", default="1.png")
    ap.add_argument("--out", default="exp3_repeat")
    ap.add_argument("--key", default="my-secret-key-2026")
    ap.add_argument("--wrong_key", default="my-secret-key-2026_wrong")
    ap.add_argument("--cr", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--iter", type=int, default=150)
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--extra", default="")
    args = ap.parse_args()

    root = Path(args.out).resolve()
    ensure_dir(root)
    extra_args = args.extra.split() if args.extra.strip() else []

    rows = []
    for r in range(1, args.N + 1):
        run_dir = root / f"rep_{r}"
        ensure_dir(run_dir)
        t0 = time.perf_counter()
        rc = run_demo(args.demo, args.img, run_dir, args.key, args.wrong_key, args.cr, args.lam, args.iter, extra_args)
        dt = time.perf_counter() - t0

        row = {"rep": r, "return_code": rc, "wall_s": dt, "run_dir": str(run_dir)}
        rep = run_dir / "report.txt"
        if rc == 0 and rep.exists():
            row.update(parse_report(rep))
        rows.append(row)
        print(f"[{r}/{args.N}] rc={rc} PSNR={row.get('PSNR')} SSIM={row.get('SSIM')}")

    write_csv(rows, root / "per_rep.csv")

    ok = [x for x in rows if x["return_code"] == 0]
    summary = []
    for k in ["PSNR","SSIM","MAE","Entropy","NPCR","UACI","enc_time_s","dec_time_s"]:
        mu, sd = mean_std(ok, k)
        summary.append({"metric": k, "mean": mu, "std": sd})
    write_csv(summary, root / "summary_mean_std.csv")

if __name__ == "__main__":
    main()
