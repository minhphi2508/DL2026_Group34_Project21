from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


REPLACEMENTS = [
    {
        "source": "HKM",
        "split": "train",
        "remove": "HKM_0074_49b70bb1c9.jpg",
        "add": "HKM_0310_4bfee08430.jpg",
        "remove_reason": "stereoscopic_double_image_unsuitable_as_single_photo_gt",
        "add_reason": "core_reaudit_replacement_historical_photo",
    },
    {
        "source": "HKM",
        "split": "train",
        "remove": "HKM_0103_9651337024.jpg",
        "add": "HKM_0538_fc7f071297.jpg",
        "remove_reason": "illustration_not_photograph",
        "add_reason": "core_reaudit_replacement_historical_photo",
    },
    {
        "source": "JOKA",
        "split": "train",
        "remove": "JOKA_0127_aae529369b.jpg",
        "add": "JOKA_0284_fe89c1f949.jpg",
        "remove_reason": "modern_documentation_photo_outside_historical_photo_domain",
        "add_reason": "core_reaudit_replacement_historical_photo",
    },
]


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=Path("dataset_v1"))
    ap.add_argument("--work-root", type=Path, default=Path("work"))
    args = ap.parse_args()

    dataset = args.dataset_root
    work = args.work_root

    core_manifest_path = dataset/"metadata"/"core_manifest.csv"
    attribution_path = dataset/"metadata"/"attribution.csv"

    # Backups are created once.
    for p in (core_manifest_path, attribution_path):
        bak = p.with_suffix(p.suffix + ".pre_reaudit.bak")
        if not bak.exists():
            shutil.copy2(p, bak)

    core_rows = read_csv(core_manifest_path)
    attribution_rows = read_csv(attribution_path)

    review_cache = {}
    for source in ("HKM", "JOKA"):
        rp = work/"review"/f"{source}_review.csv"
        bak = rp.with_suffix(rp.suffix + ".pre_reaudit.bak")
        if not bak.exists():
            shutil.copy2(rp, bak)
        review_cache[source] = read_csv(rp)

    quarantine = dataset/"quarantine_core_reaudit"
    quarantine.mkdir(parents=True, exist_ok=True)

    for item in REPLACEMENTS:
        source = item["source"]
        split = item["split"]
        remove_name = item["remove"]
        add_name = item["add"]

        old_path = dataset/"core_old"/split/source/remove_name
        new_path = dataset/"core_old"/split/source/add_name
        reserve_path = work/"candidates"/source/add_name

        if not old_path.exists():
            raise RuntimeError(f"Expected current core file missing: {old_path}")
        if new_path.exists():
            raise RuntimeError(f"Replacement already exists in core: {new_path}")
        if not reserve_path.exists():
            raise RuntimeError(f"Replacement candidate missing: {reserve_path}")

        # Quarantine removed file instead of deleting it.
        q = quarantine/source/remove_name
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_path), str(q))
        shutil.copy2(reserve_path, new_path)

        # Update work/review decision rows.
        rows = review_cache[source]
        rem = next((r for r in rows if r.get("local_file") == remove_name), None)
        add = next((r for r in rows if r.get("local_file") == add_name), None)
        if rem is None or add is None:
            raise RuntimeError(f"Could not find review rows for {remove_name} / {add_name}")
        rem["decision"] = "reject"
        rem["reason"] = item["remove_reason"]
        rem["notes"] = (rem.get("notes") or "") + "; removed_by_core_reaudit"
        add["decision"] = "core"
        add["reason"] = item["add_reason"]
        add["notes"] = (add.get("notes") or "") + f"; promoted_to_{split}_by_core_reaudit"

        # Replace core manifest row, keeping same split.
        old_idx = next(
            (i for i, r in enumerate(core_rows) if r.get("local_file") == remove_name),
            None,
        )
        if old_idx is None:
            raise RuntimeError(f"Removed item absent from core manifest: {remove_name}")

        # Build replacement from review CSV fields and match manifest schema.
        replacement = {}
        schema = list(core_rows[old_idx].keys())
        for key in schema:
            replacement[key] = add.get(key, "")
        replacement["split"] = split
        replacement["dataset_path"] = new_path.as_posix()
        replacement["dataset_sha256"] = sha256_file(new_path)
        core_rows[old_idx] = replacement

        # Replace attribution row corresponding to old core path.
        old_attr_idx = next(
            (
                i for i, r in enumerate(attribution_rows)
                if r.get("dataset_path", "").endswith("/" + remove_name)
                or r.get("dataset_path", "").endswith("\\" + remove_name)
            ),
            None,
        )
        if old_attr_idx is None:
            raise RuntimeError(f"Removed item absent from attribution CSV: {remove_name}")

        attr_schema = list(attribution_rows[old_attr_idx].keys())
        attr = {k: "" for k in attr_schema}
        attr.update({
            "dataset_path": new_path.as_posix(),
            "source": source,
            "original_id": add.get("original_id", ""),
            "title": add.get("title", ""),
            "year": add.get("year", ""),
            "photographer": add.get("photographer", ""),
            "rights": add.get("rights", ""),
            "repo_id": add.get("repo_id", ""),
            "remote_file": add.get("remote_file", ""),
        })
        attribution_rows[old_attr_idx] = attr

        print(f"[replace] {remove_name} -> {add_name} ({source}/{split})")

    # Final in-memory sanity.
    names = [r.get("local_file") for r in core_rows]
    if len(names) != 800 or len(set(names)) != 800:
        raise RuntimeError(f"Core manifest should contain 800 unique names; got {len(names)} / {len(set(names))}")

    for item in REPLACEMENTS:
        if item["remove"] in names:
            raise RuntimeError(f"Removed file still in core manifest: {item['remove']}")
        if item["add"] not in names:
            raise RuntimeError(f"Replacement missing from core manifest: {item['add']}")

    write_csv(core_manifest_path, core_rows)
    write_csv(attribution_path, attribution_rows)
    for source, rows in review_cache.items():
        write_csv(work/"review"/f"{source}_review.csv", rows)

    print("\nCORE RE-AUDIT REPLACEMENTS APPLIED")
    print("Removed files were moved to:", quarantine)
    print("Now re-run scripts/06_audit_sources.py before doing anything else.")


if __name__ == "__main__":
    main()
