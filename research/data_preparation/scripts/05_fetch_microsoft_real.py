from __future__ import annotations

import argparse
import csv
from pathlib import Path

import requests
from tqdm import tqdm

RAW_BASE = "https://raw.githubusercontent.com/microsoft/Bringing-Old-Photos-Back-to-Life/master/test_images"

FILES = {
    "old": [f"{c}.png" for c in "abcdefgh"],
    "old_w_scratch": [f"{c}.png" for c in "abcd"],
}


def get(url: str, dst: Path):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dst.write_bytes(r.content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=Path("dataset_v1"))
    args = ap.parse_args()

    out_root = args.dataset_root / "external_real" / "microsoft"
    rows = []
    for category, names in FILES.items():
        for name in tqdm(names, desc=f"Microsoft {category}"):
            url = f"{RAW_BASE}/{category}/{name}"
            dst = out_root / category / name
            get(url, dst)
            rows.append({
                "source": "Microsoft-Bringing-Old-Photos-Back-to-Life",
                "category": category,
                "file": name,
                "dataset_path": dst.as_posix(),
                "source_url": url,
                "ground_truth_available": "no",
                "evaluation_role": "qualitative_external_real_test",
            })

    meta = args.dataset_root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    with (meta / "microsoft_real_manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    (out_root / "SOURCE_NOTICE.txt").write_text(
        "Official repository sample images from:\n"
        "https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life\n"
        "Use as a small frozen qualitative real-old-photo test only.\n"
        "No paired clean ground truth is provided for these sample images.\n",
        encoding="utf-8",
    )

    print(f"Downloaded {len(rows)} Microsoft sample images to {out_root}")


if __name__ == "__main__":
    main()
