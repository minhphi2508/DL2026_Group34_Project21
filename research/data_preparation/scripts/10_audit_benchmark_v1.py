from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


SUITES=("noise","scratch","missing","age_quality","combined")
SEVS=("low","medium","high")


def arr(path,mode="RGB"):
    with Image.open(path) as im:
        return np.asarray(im.convert(mode))


def psnr(a,b):
    a=a.astype(np.float32);b=b.astype(np.float32)
    mse=float(np.mean((a-b)**2))
    if mse==0:return 99.0
    return 10*np.log10((255.0**2)/mse)


def mask_cov(path):
    a=arr(path,"L")
    vals=set(np.unique(a).tolist())
    if not vals.issubset({0,255}):
        raise AssertionError(f"Non-binary mask {path}: {sorted(vals)[:20]}")
    return float((a>0).mean())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--benchmark-root",type=Path,default=Path("benchmark_v1"))
    args=ap.parse_args()
    root=args.benchmark_root

    mf=root/"metadata"/"benchmark_manifest.csv"
    if not mf.exists():raise SystemExit(f"Missing {mf}")
    with mf.open(newline="",encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==3000,f"Expected 3000 manifest rows, got {len(rows)}"

    failures=[]
    stats=defaultdict(list)
    mask_stats=defaultdict(list)

    for split in ("val","test"):
        gts=sorted((root/split/"gt").glob("*.png"))
        if len(gts)!=100: failures.append(f"{split}: expected 100 GT, got {len(gts)}")
        for p in gts[:]:
            with Image.open(p) as im:
                if im.size!=(512,512):failures.append(f"GT wrong size: {p} {im.size}")

    for r in rows:
        inp=Path(r["input_path"]);gt=Path(r["gt_path"])
        if not inp.exists() or not gt.exists():
            failures.append(f"Missing input/gt for {r['id']} {r['split']} {r['suite']} {r['severity']}")
            continue
        with Image.open(inp) as im:
            if im.size!=(512,512):failures.append(f"Input wrong size: {inp} {im.size}")
        A=arr(inp);B=arr(gt)
        stats[(r["split"],r["suite"],r["severity"])].append(psnr(A,B))

        if r["scratch_mask_path"]:
            p=Path(r["scratch_mask_path"])
            if not p.exists():failures.append(f"Missing scratch mask {p}")
            else:mask_stats[(r["split"],"scratch",r["severity"])].append(mask_cov(p))
        if r["missing_mask_path"]:
            p=Path(r["missing_mask_path"])
            if not p.exists():failures.append(f"Missing missing mask {p}")
            else:mask_stats[(r["split"],"missing",r["severity"])].append(mask_cov(p))

    print("=== Aggregate PSNR to GT (sanity only) ===")
    for split in ("val","test"):
        for suite in SUITES:
            vals=[]
            for sev in SEVS:
                mean=float(np.mean(stats[(split,suite,sev)]))
                vals.append(mean)
                print(f"{split:4s} {suite:11s} {sev:6s}: {mean:6.2f} dB")
            # Low should generally be closer to GT than high.
            if not (vals[0] > vals[2]):
                failures.append(f"{split}/{suite}: low severity PSNR not greater than high: {vals}")

    print("\n=== Mask coverage ===")
    for split in ("val","test"):
        for kind in ("scratch","missing"):
            vals=[]
            for sev in SEVS:
                data=mask_stats[(split,kind,sev)]
                mean=float(np.mean(data)) if data else 0.0
                vals.append(mean)
                print(f"{split:4s} {kind:7s} {sev:6s}: {100*mean:6.3f}%")
            if not (vals[0] < vals[2]):
                failures.append(f"{split}/{kind}: low coverage not lower than high: {vals}")

    # Counts per cell.
    for split in ("val","test"):
        for suite in SUITES:
            for sev in SEVS:
                n=sum(1 for r in rows if r["split"]==split and r["suite"]==suite and r["severity"]==sev)
                if n!=100:failures.append(f"{split}/{suite}/{sev}: expected 100 rows got {n}")

    print("\n=== RESULT ===")
    if failures:
        print("AUDIT FAILED")
        for x in failures[:50]:print(" -",x)
        raise SystemExit(1)
    print("AUDIT PASSED")
    print("benchmark_v1 has consistent counts, 512x512 GT/input, binary masks and sensible severity ordering.")


if __name__=="__main__":
    main()
