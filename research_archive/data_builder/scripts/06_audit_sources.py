from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def all_images(path: Path):
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=Path("dataset_v1"))
    args = ap.parse_args()
    root = args.dataset_root

    failures = []

    expected = {
        ("train", "HKM"): 300,
        ("train", "JOKA"): 300,
        ("val", "HKM"): 50,
        ("val", "JOKA"): 50,
        ("test", "HKM"): 50,
        ("test", "JOKA"): 50,
    }

    print("=== CORE OLD ===")
    hash_to_split = defaultdict(set)
    total = 0
    for (split, source), n in expected.items():
        folder = root / "core_old" / split / source
        imgs = all_images(folder) if folder.exists() else []
        print(f"{split:5s} {source:4s}: {len(imgs)}")
        if len(imgs) != n:
            failures.append(f"core count {split}/{source}: expected {n}, got {len(imgs)}")
        total += len(imgs)
        for p in imgs:
            hash_to_split[sha256(p)].add(split)

    if total != 800:
        failures.append(f"core total expected 800, got {total}")

    leaks = {h: s for h, s in hash_to_split.items() if len(s) > 1}
    print(f"Core exact-hash cross-split leakage: {len(leaks)}")
    if leaks:
        failures.append(f"{len(leaks)} exact file hashes appear across multiple core splits")

    print("\n=== SYNOLD ===")
    for split, expected_count in (("train", 500), ("test", 200)):
        ins = all_images(root / "external_synthetic" / "synold" / split / "input")
        gts = all_images(root / "external_synthetic" / "synold" / split / "gt")
        print(f"{split}: input={len(ins)} gt={len(gts)}")
        if len(ins) != expected_count or len(gts) != expected_count:
            failures.append(
                f"SynOld {split}: expected {expected_count}/{expected_count}, "
                f"got {len(ins)}/{len(gts)}"
            )
        pairs = min(len(ins), len(gts))
        for i in range(pairs):
            with Image.open(ins[i]) as a, Image.open(gts[i]) as b:
                if a.size != b.size:
                    failures.append(
                        f"SynOld size mismatch: {ins[i].name} {a.size} vs {b.size}"
                    )
                    break

    print("\n=== MICROSOFT REAL ===")
    for cat, expected_count in (("old", 8), ("old_w_scratch", 4)):
        imgs = all_images(root / "external_real" / "microsoft" / cat)
        print(f"{cat}: {len(imgs)}")
        if len(imgs) != expected_count:
            failures.append(
                f"Microsoft {cat}: expected {expected_count}, got {len(imgs)}"
            )

    finna_real = all_images(root / "external_real" / "finna_damaged")
    print(f"\nFinna natural-damage qualitative set: {len(finna_real)}")

    print("\n=== RESULT ===")
    if failures:
        print("AUDIT FAILED")
        for x in failures:
            print(" -", x)
        raise SystemExit(1)
    print("AUDIT PASSED")
    print("The four-source base is consistent and ready for degradation-generator calibration.")


if __name__ == "__main__":
    main()
