from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import (
    first_photographer,
    image_metrics,
    read_jsonl,
    rights_string,
    safe_year,
    sanitize_token,
    sha256_file,
    technical_ok,
)

SOURCES = {
    "HKM": {
        "repo_id": "NatLibFi/Finna-HKM-images",
        "max_year": 1917,
    },
    "JOKA": {
        "repo_id": "NatLibFi/Finna-JOKA-images",
        "max_year": 1940,
    },
}


def eligible_rows(metadata_path: Path, max_year: int):
    out = []
    seen_ids = set()
    for idx, row in enumerate(read_jsonl(metadata_path)):
        rid = str(row.get("id") or "").strip()
        file_name = str(row.get("file_name") or "").strip()
        if not rid or not file_name or rid in seen_ids:
            continue
        year = safe_year(row.get("year"))
        if year is not None and year > max_year:
            continue
        rights = rights_string(row).lower().replace("-", " ")
        if "cc by 4.0" not in rights and "cc by 4" not in rights:
            continue
        row["_metadata_index"] = idx
        row["_parsed_year"] = year
        out.append(row)
        seen_ids.add(rid)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", type=Path, default=Path("work"))
    ap.add_argument("--per-source", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument(
        "--max-per-photographer",
        type=int,
        default=40,
        help="Soft diversity cap for known photographer names. Unknown authors are not capped.",
    )
    args = ap.parse_args()

    review_dir = args.work_root / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    for s_i, (source, cfg) in enumerate(SOURCES.items()):
        print(f"\n=== {source}: {cfg['repo_id']} ===")
        meta_path = Path(
            hf_hub_download(
                repo_id=cfg["repo_id"],
                filename="metadata.jsonl",
                repo_type="dataset",
            )
        )
        rows = eligible_rows(meta_path, cfg["max_year"])
        print(f"Eligible metadata rows: {len(rows)}")

        rng = random.Random(args.seed + s_i * 100003)
        rng.shuffle(rows)

        out_dir = args.work_root / "candidates" / source
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest_rows = []
        rejected_rows = []
        photographer_counts = Counter()

        pbar = tqdm(total=args.per_source, desc=f"Downloading {source}")
        for row in rows:
            if len(manifest_rows) >= args.per_source:
                break

            photographer = first_photographer(row)
            if (
                photographer
                and photographer_counts[photographer] >= args.max_per_photographer
            ):
                continue

            remote_name = str(row["file_name"])
            try:
                cached = Path(
                    hf_hub_download(
                        repo_id=cfg["repo_id"],
                        filename=remote_name,
                        repo_type="dataset",
                    )
                )
                metrics = image_metrics(cached)
                ok, reason = technical_ok(metrics)
                if not ok:
                    rejected_rows.append(
                        {
                            "source": source,
                            "original_id": row.get("id", ""),
                            "remote_file": remote_name,
                            "reason": reason,
                            **metrics,
                        }
                    )
                    continue

                rid = str(row.get("id", ""))
                short_hash = hashlib.sha256(rid.encode("utf-8")).hexdigest()[:10]
                suffix = cached.suffix.lower() if cached.suffix else ".jpg"
                local_name = f"{source}_{len(manifest_rows)+1:04d}_{short_hash}{suffix}"
                dst = out_dir / local_name
                shutil.copy2(cached, dst)

                rec = {
                    "source": source,
                    "local_file": local_name,
                    "original_id": rid,
                    "title": str(row.get("title") or ""),
                    "year": row.get("_parsed_year") or "",
                    "photographer": photographer,
                    "rights": rights_string(row),
                    "repo_id": cfg["repo_id"],
                    "remote_file": remote_name,
                    "sha256": sha256_file(dst),
                    **metrics,
                }
                manifest_rows.append(rec)
                if photographer:
                    photographer_counts[photographer] += 1
                pbar.update(1)
            except Exception as e:
                rejected_rows.append(
                    {
                        "source": source,
                        "original_id": row.get("id", ""),
                        "remote_file": remote_name,
                        "reason": f"download_or_decode_error:{type(e).__name__}:{e}",
                    }
                )
        pbar.close()

        if len(manifest_rows) < args.per_source:
            raise SystemExit(
                f"{source}: only obtained {len(manifest_rows)} technically usable images "
                f"but target was {args.per_source}. Re-run with looser source diversity or inspect failures."
            )

        manifest_path = args.work_root / f"{source}_candidates_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader()
            w.writerows(manifest_rows)

        review_path = review_dir / f"{source}_review.csv"
        review_fields = list(manifest_rows[0].keys()) + [
            "decision",
            "reason",
            "notes",
        ]
        with review_path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=review_fields)
            w.writeheader()
            for rec in manifest_rows:
                w.writerow({**rec, "decision": "", "reason": "", "notes": ""})

        rejected_path = args.work_root / f"{source}_automatic_rejects.csv"
        if rejected_rows:
            fields = sorted({k for r in rejected_rows for k in r.keys()})
            with rejected_path.open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rejected_rows)

        print(f"Saved {len(manifest_rows)} candidates to {out_dir}")
        print(f"Review file: {review_path}")

    print("\nCandidate stage complete.")
    print("Next: python scripts/02_make_contact_sheets.py --work-root work")


if __name__ == "__main__":
    main()
