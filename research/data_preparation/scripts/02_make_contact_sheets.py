from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def get_font(size=17):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def fit_thumb(im: Image.Image, size):
    im = im.convert("RGB")
    return ImageOps.contain(im, size, method=Image.Resampling.LANCZOS)


def make_sheet(records, image_dir: Path, out_path: Path, cols=5, rows=4):
    cell_w, cell_h = 300, 250
    image_h = 205
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = get_font(16)
    small = get_font(13)

    for idx, rec in enumerate(records):
        r, c = divmod(idx, cols)
        x0, y0 = c * cell_w, r * cell_h
        p = image_dir / rec["local_file"]
        with Image.open(p) as im:
            thumb = fit_thumb(im, (cell_w - 12, image_h - 8))
        x = x0 + (cell_w - thumb.width) // 2
        y = y0 + 4 + (image_h - thumb.height) // 2
        canvas.paste(thumb, (x, y))
        draw.rectangle((x0, y0, x0 + cell_w - 1, y0 + cell_h - 1), outline="gray")
        label = rec["local_file"]
        year = rec.get("year", "")
        draw.text((x0 + 5, y0 + image_h + 2), label, fill="black", font=small)
        draw.text((x0 + 5, y0 + image_h + 20), f"year={year}", fill="black", font=small)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", type=Path, default=Path("work"))
    ap.add_argument("--per-sheet", type=int, default=20)
    args = ap.parse_args()

    for source in ("HKM", "JOKA"):
        review_path = args.work_root / "review" / f"{source}_review.csv"
        if not review_path.exists():
            raise SystemExit(f"Missing {review_path}; run 01_fetch_finna_candidates.py first.")
        with review_path.open(newline="", encoding="utf-8-sig") as f:
            records = list(csv.DictReader(f))

        image_dir = args.work_root / "candidates" / source
        out_dir = args.work_root / "contact_sheets" / source
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in range(0, len(records), args.per_sheet):
            chunk = records[i:i+args.per_sheet]
            out = out_dir / f"{source}_sheet_{i//args.per_sheet+1:03d}.jpg"
            make_sheet(chunk, image_dir, out)

        print(f"{source}: {math.ceil(len(records)/args.per_sheet)} contact sheets -> {out_dir}")

    print("\nReview the sheets, then edit work/review/*_review.csv.")
    print("Use decisions: core / real_damage / reject.")


if __name__ == "__main__":
    main()
