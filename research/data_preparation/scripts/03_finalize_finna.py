from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import sha256_file

CORE_TARGET = 400
SPLITS = {"train": 300, "val": 50, "test": 50}
VALID_DECISIONS = {"core", "real_damage", "reject", ""}


def read_review(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", type=Path, default=Path("work"))
    ap.add_argument("--dataset-root", type=Path, default=Path("dataset_v1"))
    ap.add_argument("--seed", type=int, default=20260929)
    args = ap.parse_args()

    all_core = []
    all_real = []
    attribution = []

    for s_i, source in enumerate(("HKM", "JOKA")):
        review_path = args.work_root / "review" / f"{source}_review.csv"
        rows = read_review(review_path)

        bad = sorted({(r.get("decision") or "").strip().lower() for r in rows} - VALID_DECISIONS)
        if bad:
            raise SystemExit(f"{source}: invalid review decisions: {bad}")

        core = [r for r in rows if (r.get("decision") or "").strip().lower() == "core"]
        real = [r for r in rows if (r.get("decision") or "").strip().lower() == "real_damage"]

        if len(core) != CORE_TARGET:
            raise SystemExit(
                f"{source}: mark exactly {CORE_TARGET} rows as 'core'; currently {len(core)}."
            )

        rng = random.Random(args.seed + s_i * 100003)
        rng.shuffle(core)

        cursor = 0
        for split, count in SPLITS.items():
            part = core[cursor:cursor+count]
            cursor += count
            for rec in part:
                src = args.work_root / "candidates" / source / rec["local_file"]
                dst = args.dataset_root / "core_old" / split / source / rec["local_file"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                out = dict(rec)
                out["split"] = split
                out["dataset_path"] = dst.as_posix()
                out["dataset_sha256"] = sha256_file(dst)
                all_core.append(out)
                attribution.append({
                    "dataset_path": dst.as_posix(),
                    "source": source,
                    "original_id": rec.get("original_id",""),
                    "title": rec.get("title",""),
                    "year": rec.get("year",""),
                    "photographer": rec.get("photographer",""),
                    "rights": rec.get("rights",""),
                    "repo_id": rec.get("repo_id",""),
                    "remote_file": rec.get("remote_file",""),
                })

        for rec in real:
            src = args.work_root / "candidates" / source / rec["local_file"]
            dst = args.dataset_root / "external_real" / "finna_damaged" / source / rec["local_file"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            out = dict(rec)
            out["dataset_path"] = dst.as_posix()
            out["dataset_sha256"] = sha256_file(dst)
            all_real.append(out)
            attribution.append({
                "dataset_path": dst.as_posix(),
                "source": source,
                "original_id": rec.get("original_id",""),
                "title": rec.get("title",""),
                "year": rec.get("year",""),
                "photographer": rec.get("photographer",""),
                "rights": rec.get("rights",""),
                "repo_id": rec.get("repo_id",""),
                "remote_file": rec.get("remote_file",""),
            })

    write_csv(args.dataset_root / "metadata" / "core_manifest.csv", all_core)
    write_csv(args.dataset_root / "metadata" / "real_damage_manifest.csv", all_real)
    write_csv(args.dataset_root / "metadata" / "attribution.csv", attribution)

    print("Finna core frozen.")
    print(f"Core images: {len(all_core)}")
    print(f"Natural-damage holdout: {len(all_real)}")
    print("Expected core split totals: train=600 val=100 test=100")


if __name__ == "__main__":
    main()
