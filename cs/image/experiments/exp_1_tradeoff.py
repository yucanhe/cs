# exp_1_tradeoff.py
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

from exp_common import ensure_dir, parse_report, write_csv, plot_xy

def run_demo(demo, img, out_dir, key, wrong_key, cr, lam, it, extra_args):
    cmd = [sys.executable, demo,
           "--img", img, "--out", str(out_dir),
           "--key", key, "--wrong_key", wrong_key,
           "--cr", str(cr), "--lam", str(lam), "--iter", str(it)]
    cmd += extra_args
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (out_dir / "console.log").write_text(p.stdout, encoding="utf-8", errors="ignore")
    return p.returncode

def main():
    ap = argparse.ArgumentParser()

    # 默认 demo 路径：始终指向 image/demo/demo3.py（与运行目录无关）
    demo_default = str((Path(__file__).resolve().parent.parent / "demo" / "demo3.py"))
    ap.add_argument("--demo", default=demo_default)
    ap.add_argument("--img", default="1.png")
    ap.add_argument("--out", default="exp1_tradeoff")
    ap.add_argument("--key", default="my-secret-key-2026")
    ap.add_argument("--wrong_key", default="my-secret-key-2026_wrong")
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--iter", type=int, default=150)
    ap.add_argument("--crs", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--extra", default="", help="extra flags passed to demo (e.g. '--lam_auto')")
    args = ap.parse_args()

    root = Path(args.out).resolve()
    ensure_dir(root)

    crs = [float(x) for x in args.crs.split(",")]
    extra_args = args.extra.split() if args.extra.strip() else []

    rows = []
    for idx, cr in enumerate(crs, 1):
        run_dir = root / f"cr_{cr:.2f}"
        ensure_dir(run_dir)
        t0 = time.perf_counter()
        rc = run_demo(args.demo, args.img, run_dir, args.key, args.wrong_key, cr, args.lam, args.iter, extra_args)
        dt = time.perf_counter() - t0

        row = {"cr": cr, "return_code": rc, "wall_s": dt, "run_dir": str(run_dir)}
        rep = run_dir / "report.txt"
        if rc == 0 and rep.exists():
            row.update(parse_report(rep))
        rows.append(row)
        print(f"[{idx}/{len(crs)}] cr={cr:.2f} rc={rc} PSNR={row.get('PSNR')} SSIM={row.get('SSIM')}")

    write_csv(rows, root / "results.csv")

    ok = [r for r in rows if r["return_code"] == 0 and r.get("PSNR") is not None]
    ok.sort(key=lambda r: r["cr"])
    crx = [r["cr"] for r in ok]
    plot_xy(crx, [r["PSNR"] for r in ok], root / "psnr_vs_cr.png", "cr", "PSNR", "PSNR vs cr")
    plot_xy(crx, [r["SSIM"] for r in ok], root / "ssim_vs_cr.png", "cr", "SSIM", "SSIM vs cr")
    plot_xy(crx, [r["NPCR"] for r in ok], root / "npcr_vs_cr.png", "cr", "NPCR(%)", "NPCR vs cr")
    plot_xy(crx, [r["UACI"] for r in ok], root / "uaci_vs_cr.png", "cr", "UACI(%)", "UACI vs cr")

if __name__ == "__main__":
    main()
