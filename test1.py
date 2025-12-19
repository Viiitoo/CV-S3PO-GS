#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

def parse_indices_from_dir(img_dir: Path):
    # grab any file that contains a number, use that number as frame index
    files = sorted([p for p in img_dir.iterdir() if p.is_file()])
    idxs = []
    for p in files:
        m = re.search(r"(\d+)", p.stem)
        if m:
            idxs.append(int(m.group(1)))
    if not idxs:
        raise SystemExit(f"No numbered files found in: {img_dir}")
    idxs = sorted(set(idxs))
    return idxs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="output calib dir, e.g. .../train/calib")
    ap.add_argument("--num_frames", type=int, default=None, help="generate 0..num_frames-1")
    ap.add_argument("--img_dir", default=None, help="optional: scan indices from this dir (rgb/depth)")
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--fx", type=float, required=True)
    ap.add_argument("--fy", type=float, required=True)
    ap.add_argument("--cx", type=float, required=True)
    ap.add_argument("--cy", type=float, required=True)
    ap.add_argument("--k1", type=float, required=True)
    ap.add_argument("--k2", type=float, required=True)
    ap.add_argument("--p1", type=float, required=True)
    ap.add_argument("--p2", type=float, required=True)
    ap.add_argument("--k3", type=float, required=True)

    ap.add_argument("--Tr_velo_to_cam", nargs=12, type=float, default=[
        1.0,0.0,0.0,0.0,
        0.0,1.0,0.0,0.0,
        0.0,0.0,1.0,0.0
    ], help="12 floats row-major 3x4 (default identity)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.img_dir:
        indices = parse_indices_from_dir(Path(args.img_dir))
    else:
        if args.num_frames is None:
            raise SystemExit("Provide either --num_frames or --img_dir")
        indices = list(range(args.num_frames))

    Tr = " ".join(f"{v:.10g}" for v in args.Tr_velo_to_cam)
    line1 = f"Tr_velo_to_cam: {Tr}\n"
    line2 = f"fx: {args.fx:.15g} fy: {args.fy:.15g} cx: {args.cx:.15g} cy: {args.cy:.15g}\n"
    line3 = f"k1: {args.k1:.15g} k2: {args.k2:.15g} p1: {args.p1:.15g} p2: {args.p2:.15g} k3: {args.k3:.15g}\n"

    written = 0
    skipped = 0
    for idx in indices:
        fp = outdir / f"{idx:06d}.txt"
        if fp.exists() and not args.overwrite:
            skipped += 1
            continue
        fp.write_text(line1 + line2 + line3)
        written += 1

    print(f"[DONE] written={written}, skipped={skipped}, outdir={outdir}")

if __name__ == "__main__":
    main()
