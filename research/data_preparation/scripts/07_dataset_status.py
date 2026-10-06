from __future__ import annotations

import argparse
import csv
from pathlib import Path


def count_rows(path: Path):
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8-sig") as f:
        return sum(1 for _ in csv.DictReader(f))


def count_decisions(path: Path):
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            d = (r.get("decision") or "").strip().lower() or "(blank)"
            out[d] = out.get(d, 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", type=Path, default=Path("work"))
    ap.add_argument("--dataset-root", type=Path, default=Path("dataset_v1"))
    args = ap.parse_args()

    print("Finna candidate/review status")
    for s in ("HKM", "JOKA"):
        p = args.work_root / "review" / f"{s}_review.csv"
        print(f"  {s}: {count_decisions(p)}")

    print("\nFrozen metadata")
    for name in (
        "core_manifest.csv",
        "real_damage_manifest.csv",
        "synold_manifest.csv",
        "microsoft_real_manifest.csv",
    ):
        p = args.dataset_root / "metadata" / name
        print(f"  {name}: {count_rows(p)} rows")


if __name__ == "__main__":
    main()
