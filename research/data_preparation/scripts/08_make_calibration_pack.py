from __future__ import annotations

import argparse
import csv
import random
import shutil
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def all_images(path: Path):
    if not path.exists():
        return []
    return sorted(
        p for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def get_font(size: int):
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def preview_image(src: Path, dst: Path, max_side: int = 1600):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            im = im.resize(
                (max(1, round(w * scale)), max(1, round(h * scale))),
                Image.Resampling.LANCZOS,
            )
        im.save(dst, format="JPEG", quality=94, subsampling=0)


def pair_preview(inp: Path, gt: Path, dst: Path, max_h: int = 700):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(inp) as a, Image.open(gt) as b:
        a = a.convert("RGB")
        b = b.convert("RGB")
        scale = min(1.0, max_h / max(a.height, b.height))
        if scale < 1.0:
            a = a.resize((round(a.width * scale), round(a.height * scale)), Image.Resampling.LANCZOS)
            b = b.resize((round(b.width * scale), round(b.height * scale)), Image.Resampling.LANCZOS)
        gap = 12
        canvas = Image.new("RGB", (a.width + b.width + gap, max(a.height, b.height) + 42), "white")
        canvas.paste(a, (0, 42))
        canvas.paste(b, (a.width + gap, 42))
        d = ImageDraw.Draw(canvas)
        f = get_font(18)
        d.text((6, 8), "DAMAGED / INPUT", fill="black", font=f)
        d.text((a.width + gap + 6, 8), "CLEAN / GT", fill="black", font=f)
        canvas.save(dst, format="JPEG", quality=94, subsampling=0)


def make_contact_sheet(items, out_path: Path, cols=4, rows=3, cell_w=410, cell_h=330):
    if not items:
        return
    per = cols * rows
    font = get_font(15)
    for page_idx in range(0, len(items), per):
        chunk = items[page_idx:page_idx+per]
        canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
        d = ImageDraw.Draw(canvas)
        for i, (img_path, label) in enumerate(chunk):
            rr, cc = divmod(i, cols)
            x0, y0 = cc * cell_w, rr * cell_h
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                thumb = ImageOps.contain(im, (cell_w - 14, cell_h - 50), Image.Resampling.LANCZOS)
            x = x0 + (cell_w - thumb.width) // 2
            y = y0 + 4
            canvas.paste(thumb, (x, y))
            d.rectangle((x0, y0, x0 + cell_w - 1, y0 + cell_h - 1), outline="gray")
            text = label if len(label) <= 52 else label[:49] + "..."
            d.text((x0 + 5, y0 + cell_h - 38), text, fill="black", font=font)
        suffix = f"_{page_idx // per + 1:02d}"
        p = out_path.with_name(out_path.stem + suffix + out_path.suffix)
        p.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(p, quality=92, subsampling=0)


def choose_balanced_core(root: Path, source: str, n: int, rng: random.Random):
    pool = []
    for split in ("train", "val"):
        pool.extend(all_images(root / "core_old" / split / source))
    if len(pool) < n:
        raise RuntimeError(f"Not enough {source} train/val images: {len(pool)} < {n}")
    rng.shuffle(pool)
    return pool[:n]


def find_synold_pair_dirs(root: Path):
    inp = root / "external_synthetic" / "synold" / "train" / "input"
    gt = root / "external_synthetic" / "synold" / "train" / "gt"
    return inp, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=Path("dataset_v1"))
    ap.add_argument("--output-root", type=Path, default=Path("calibration_pack"))
    ap.add_argument("--seed", type=int, default=20260930)
    args = ap.parse_args()

    root = args.dataset_root
    out = args.output_root
    if out.exists():
        shutil.rmtree(out)
    (out / "contact_sheets").mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    manifest = []

    # 1) Clean-ish historical core: 24 HKM + 24 JOKA, train/val only.
    core_preview_items = []
    for source in ("HKM", "JOKA"):
        selected = choose_balanced_core(root, source, 24, rng)
        for i, src in enumerate(selected, 1):
            dst = out / "previews" / "core_cleanish" / f"{source}_{i:02d}_{src.stem}.jpg"
            preview_image(src, dst)
            core_preview_items.append((dst, f"{source} | {src.name}"))
            manifest.append({
                "group": "core_cleanish",
                "source": source,
                "role": "historical_cleanish_reference",
                "original_path": src.as_posix(),
                "preview_path": dst.as_posix(),
                "notes": "sampled from train/val only; frozen test excluded",
            })
    make_contact_sheet(core_preview_items, out / "contact_sheets" / "core_cleanish.jpg")

    # 2) All real-damage examples.
    real_paths = []
    finna_real = root / "external_real" / "finna_damaged"
    for p in all_images(finna_real):
        real_paths.append(("FINNA_REAL_DAMAGE", p))
    ms_root = root / "external_real" / "microsoft"
    for category in ("old", "old_w_scratch"):
        for p in all_images(ms_root / category):
            real_paths.append((f"MICROSOFT_{category.upper()}", p))

    real_preview_items = []
    for i, (source, src) in enumerate(real_paths, 1):
        dst = out / "previews" / "real_damage" / f"{i:02d}_{source}_{src.name}.jpg"
        preview_image(src, dst, max_side=1800)
        real_preview_items.append((dst, f"{source} | {src.name}"))
        manifest.append({
            "group": "real_damage",
            "source": source,
            "role": "real_world_damage_calibration",
            "original_path": src.as_posix(),
            "preview_path": dst.as_posix(),
            "notes": "qualitative/no-reference only",
        })
    make_contact_sheet(real_preview_items, out / "contact_sheets" / "real_damage.jpg")

    # 3) SynOld train pairs.
    syn_inp, syn_gt = find_synold_pair_dirs(root)
    syn_inputs = all_images(syn_inp)
    if len(syn_inputs) < 40:
        raise RuntimeError(f"Expected >=40 SynOld train inputs, found {len(syn_inputs)}")
    rng.shuffle(syn_inputs)
    syn_inputs = syn_inputs[:40]

    pair_preview_items = []
    for i, inp in enumerate(syn_inputs, 1):
        gt = syn_gt / inp.name
        if not gt.exists():
            raise RuntimeError(f"Missing SynOld GT for {inp.name}")
        dst = out / "previews" / "synold_pairs" / f"{i:02d}_{inp.stem}.jpg"
        pair_preview(inp, gt, dst)
        pair_preview_items.append((dst, f"SynOld train pair | {inp.name}"))
        manifest.append({
            "group": "synold_pair",
            "source": "SynOld",
            "role": "scratch_style_calibration",
            "original_path": inp.as_posix(),
            "preview_path": dst.as_posix(),
            "notes": f"GT={gt.as_posix()}",
        })
    make_contact_sheet(pair_preview_items, out / "contact_sheets" / "synold_pairs.jpg")

    # Manifest
    mf = out / "calibration_manifest.csv"
    with mf.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    notes = f"""Calibration pack created with seed {args.seed}.

Counts:
- core_cleanish historical references: {sum(1 for x in manifest if x['group']=='core_cleanish')}
- real damaged historical examples: {sum(1 for x in manifest if x['group']=='real_damage')}
- SynOld paired scratch examples: {sum(1 for x in manifest if x['group']=='synold_pair')}

The core historical sample is drawn only from train/val.
No `core_old/test` image was used for degradation calibration.
"""
    (out / "CALIBRATION_NOTES.txt").write_text(notes, encoding="utf-8")

    # Zip output for upload/review.
    zip_path = out.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(out.parent))

    print("CALIBRATION PACK CREATED")
    print(notes)
    print(f"ZIP: {zip_path}")


if __name__ == "__main__":
    main()
