from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_year(value: Any) -> int | None:
    if value is None:
        return None
    m = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", str(value))
    return int(m.group(1)) if m else None


def first_photographer(row: dict) -> str:
    presenters = row.get("presenters") or []
    if isinstance(presenters, list):
        for x in presenters:
            if isinstance(x, dict) and x.get("name"):
                return str(x["name"]).strip()
    authors = row.get("nonPresenterAuthors") or []
    if isinstance(authors, list):
        for x in authors:
            if isinstance(x, dict) and x.get("name"):
                return str(x["name"]).strip()
            if isinstance(x, str) and x.strip():
                return x.strip()
    return ""


def rights_string(row: dict) -> str:
    r = row.get("imageRights") or {}
    if isinstance(r, dict):
        return str(r.get("copyright") or "")
    return str(r)


def image_metrics(path: Path) -> dict:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        arr = np.asarray(im.resize((min(w, 512), min(h, 512))), dtype=np.float32)
        gray = arr.mean(axis=2)
        dx = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
        dy = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
        return {
            "width": int(w),
            "height": int(h),
            "short_side": int(min(w, h)),
            "long_side": int(max(w, h)),
            "aspect_ratio": float(w / h),
            "mean_luma": float(gray.mean()),
            "luma_std": float(gray.std()),
            "gradient_mean": float((dx + dy) / 2.0),
        }


def technical_ok(metrics: dict) -> tuple[bool, str]:
    if metrics["short_side"] < 300:
        return False, "short_side<300"
    if metrics["long_side"] < 500:
        return False, "long_side<500"
    if not (0.40 <= metrics["aspect_ratio"] <= 2.50):
        return False, "extreme_aspect_ratio"
    if metrics["luma_std"] < 8.0:
        return False, "near_blank_or_extremely_low_contrast"
    return True, ""


def sanitize_token(s: str, max_len: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())
    return s[:max_len].strip("_") or "item"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Bad JSONL at {path}:{line_no}: {e}") from e
