from __future__ import annotations

import argparse
import csv
import io
import shutil
import tempfile
import zipfile
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

ARCHIVE_URL = "https://github.com/wushunshun/SynOld/archive/refs/heads/main.zip"


def download(url: str, dst: Path):
    if dst.exists():
        return
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dst, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dst.name
        ) as p:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    p.update(len(chunk))


def split_pair(src: Path, input_dst: Path, gt_dst: Path):
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        if w < 2:
            raise ValueError(f"Invalid pair width: {src} {im.size}")
        mid = w // 2
        # Verified on repository sample train/1.jpg:
        # LEFT = synthetic scratched/damaged input, RIGHT = clean GT.
        damaged = im.crop((0, 0, mid, h))
        gt = im.crop((mid, 0, w, h))
        if damaged.size != gt.size:
            raise ValueError(f"Unequal pair halves: {src} {damaged.size} vs {gt.size}")
        input_dst.parent.mkdir(parents=True, exist_ok=True)
        gt_dst.parent.mkdir(parents=True, exist_ok=True)
        damaged.save(input_dst, quality=95)
        gt.save(gt_dst, quality=95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=Path("dataset_v1"))
    args = ap.parse_args()

    out_root = args.dataset_root / "external_synthetic" / "synold"
    out_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        z = td / "synold.zip"
        download(ARCHIVE_URL, z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(td)
        repo = td / "SynOld-main"
        if not repo.exists():
            matches = list(td.glob("SynOld-*"))
            if not matches:
                raise SystemExit("Could not locate extracted SynOld repository.")
            repo = matches[0]

        manifest = []
        for split, expected in (("train", 500), ("test", 200)):
            files = sorted((repo / split).glob("*.jpg"), key=lambda p: int(p.stem))
            if len(files) != expected:
                print(f"WARNING: {split} expected {expected} source pairs but found {len(files)}")
            for p in tqdm(files, desc=f"SynOld {split}"):
                name = f"{int(p.stem):04d}.png"
                inp = out_root / split / "input" / name
                gt = out_root / split / "gt" / name
                split_pair(p, inp, gt)
                with Image.open(inp) as a, Image.open(gt) as b:
                    manifest.append({
                        "source": "SynOld",
                        "official_split": split,
                        "id": p.stem,
                        "input_path": inp.as_posix(),
                        "gt_path": gt.as_posix(),
                        "width": a.width,
                        "height": a.height,
                        "pair_layout": "left_damaged_right_gt",
                    })

        meta = args.dataset_root / "metadata"
        meta.mkdir(parents=True, exist_ok=True)
        with (meta / "synold_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
            w.writeheader()
            w.writerows(manifest)

    notice = out_root / "SOURCE_NOTICE.txt"
    notice.write_text(
        "Source: https://github.com/wushunshun/SynOld\n"
        "README states 500 training pairs and 200 test pairs for old-photo scratch restoration.\n"
        "Repository metadata did not declare a license when dataset_v1 was designed.\n"
        "Use internally for the class project unless reuse terms are clarified.\n",
        encoding="utf-8",
    )
    print(f"SynOld prepared at {out_root}")


if __name__ == "__main__":
    main()
