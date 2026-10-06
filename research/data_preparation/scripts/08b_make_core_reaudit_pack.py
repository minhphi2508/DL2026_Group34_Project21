from __future__ import annotations

import argparse
import csv
import re
import shutil
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def font(size):
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def read_csv(path: Path):
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


def parse_rank(notes: str) -> int:
    m = re.search(r"review_rank=(\d+)", notes or "")
    return int(m.group(1)) if m else 10**9


def contain(im, size):
    return ImageOps.contain(im.convert("RGB"), size, Image.Resampling.LANCZOS)


def make_sheets(rows, image_path_fn, out_dir: Path, prefix: str, per_sheet=12):
    cols, rows_n = 4, 3
    cell_w, cell_h = 470, 390
    img_h = 315
    f1 = font(14)
    f2 = font(12)
    out_dir.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(rows), per_sheet):
        chunk = rows[start:start+per_sheet]
        canvas = Image.new("RGB", (cols*cell_w, rows_n*cell_h), "white")
        d = ImageDraw.Draw(canvas)

        for i, rec in enumerate(chunk):
            rr, cc = divmod(i, cols)
            x0, y0 = cc*cell_w, rr*cell_h
            p = image_path_fn(rec)
            try:
                with Image.open(p) as im:
                    th = contain(im, (cell_w-12, img_h-8))
            except Exception:
                th = Image.new("RGB", (cell_w-30, img_h-30), "#dddddd")
            x = x0 + (cell_w-th.width)//2
            y = y0 + 4 + (img_h-th.height)//2
            canvas.paste(th, (x,y))
            d.rectangle((x0,y0,x0+cell_w-1,y0+cell_h-1), outline="gray")

            name = rec.get("local_file","")
            year = rec.get("year","")
            title = (rec.get("title","") or "").replace("\n"," ").strip()
            if len(title) > 55:
                title = title[:52] + "..."
            d.text((x0+5, y0+img_h+3), name, fill="black", font=f1)
            d.text((x0+5, y0+img_h+23), f"year={year} | {title}", fill="black", font=f2)

        page = start//per_sheet + 1
        canvas.save(out_dir/f"{prefix}_{page:03d}.jpg", quality=93, subsampling=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=Path("dataset_v1"))
    ap.add_argument("--work-root", type=Path, default=Path("work"))
    ap.add_argument("--output-root", type=Path, default=Path("core_reaudit_pack"))
    ap.add_argument("--reserve-per-source", type=int, default=100)
    args = ap.parse_args()

    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True)

    core_manifest = read_csv(args.dataset_root/"metadata"/"core_manifest.csv")
    core_by_source = {"HKM": [], "JOKA": []}
    for r in core_manifest:
        src = r.get("source")
        if src in core_by_source:
            core_by_source[src].append(r)

    # Current core: sort by filename so review references are stable.
    review_core_rows = []
    for source in ("HKM","JOKA"):
        core_by_source[source].sort(key=lambda r: r.get("local_file",""))
        make_sheets(
            core_by_source[source],
            lambda r, s=source: args.dataset_root/"core_old"/r["split"]/s/r["local_file"],
            args.output_root/"core_sheets"/source,
            f"{source}_core",
        )
        for r in core_by_source[source]:
            review_core_rows.append({
                "source": source,
                "split": r.get("split",""),
                "local_file": r.get("local_file",""),
                "year": r.get("year",""),
                "title": r.get("title",""),
                "photographer": r.get("photographer",""),
                "current_decision": "core",
                "reaudit_decision": "",
                "reaudit_reason": "",
            })

    # Reserve candidates: previously rejected only because they fell below the 400-core quota,
    # not because they were explicitly flagged as nonstandard.
    reserve_rows_all = []
    for source in ("HKM","JOKA"):
        review_path = args.work_root/"review"/f"{source}_review.csv"
        rows = read_csv(review_path)
        reserve = [
            r for r in rows
            if (r.get("decision") or "").strip().lower() == "reject"
            and (r.get("reason") or "").strip() == "not_selected_after_quality_and_visual_review"
        ]
        reserve.sort(key=lambda r: parse_rank(r.get("notes","")))
        reserve = reserve[:args.reserve_per_source]

        make_sheets(
            reserve,
            lambda r, s=source: args.work_root/"candidates"/s/r["local_file"],
            args.output_root/"reserve_sheets"/source,
            f"{source}_reserve",
        )

        for r in reserve:
            reserve_rows_all.append({
                "source": source,
                "local_file": r.get("local_file",""),
                "year": r.get("year",""),
                "title": r.get("title",""),
                "photographer": r.get("photographer",""),
                "previous_reason": r.get("reason",""),
                "previous_notes": r.get("notes",""),
                "reserve_decision": "",
                "reserve_notes": "",
            })

    write_csv(args.output_root/"core_manifest_for_review.csv", review_core_rows)
    write_csv(args.output_root/"reserve_manifest_for_review.csv", reserve_rows_all)

    notes = f"""CORE RE-AUDIT PACK

Current core:
- HKM: {len(core_by_source['HKM'])}
- JOKA: {len(core_by_source['JOKA'])}
- Total: {len(review_core_rows)}

Reserve pool:
- Total: {len(reserve_rows_all)}
- Target: {args.reserve_per_source} per source where available

Known examples that MUST NOT remain in core:
- HKM_0103_9651337024.jpg -> illustration/drawing
- JOKA_0127_aae529369b.jpg -> modern-looking colour object photograph, unsuitable old-photo GT domain

Do not modify train/val/test until the re-audit decisions and replacements are finalized.
"""
    (args.output_root/"README_REVIEW.txt").write_text(notes, encoding="utf-8")

    zpath = args.output_root.with_suffix(".zip")
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in args.output_root.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(args.output_root.parent))

    print("CORE RE-AUDIT PACK CREATED")
    print(notes)
    print("ZIP:", zpath)


if __name__ == "__main__":
    main()
