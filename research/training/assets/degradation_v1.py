from __future__ import annotations

from io import BytesIO
from pathlib import Path
import hashlib
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


SUITES = ("noise", "scratch", "missing", "age_quality", "combined")
SEVERITIES = ("low", "medium", "high")


DEFAULT_CONFIG = {
    "noise": {
        "low": dict(sigma_min=2.5, sigma_max=5.0, chroma_scale=.12, speckle_prob=.00010),
        "medium": dict(sigma_min=5.0, sigma_max=9.0, chroma_scale=.18, speckle_prob=.00030),
        "high": dict(sigma_min=9.0, sigma_max=16.0, chroma_scale=.25, speckle_prob=.00070),
    },
    "scratch": {
        "low": dict(count_min=1, count_max=2, width_min=1, width_max=2, alpha_min=.28, alpha_max=.50,
                    branch_prob=.04, speck_count_min=1, speck_count_max=8, scuff_clusters=0),
        "medium": dict(count_min=2, count_max=5, width_min=1, width_max=3, alpha_min=.35, alpha_max=.68,
                       branch_prob=.16, speck_count_min=8, speck_count_max=28, scuff_clusters=1),
        "high": dict(count_min=5, count_max=10, width_min=1, width_max=4, alpha_min=.42, alpha_max=.82,
                     branch_prob=.32, speck_count_min=24, speck_count_max=70, scuff_clusters=1),
    },
    "missing": {
        "low": dict(target_coverage_min=.003, target_coverage_max=.015, stroke_width_min=8, stroke_width_max=20, edge_tear_prob=.20),
        "medium": dict(target_coverage_min=.015, target_coverage_max=.050, stroke_width_min=14, stroke_width_max=36, edge_tear_prob=.45),
        "high": dict(target_coverage_min=.050, target_coverage_max=.120, stroke_width_min=24, stroke_width_max=58, edge_tear_prob=.70),
    },
    "age_quality": {
        "low": dict(blur_min=0.0, blur_max=.35, downscale_min=1.0, downscale_max=1.25, jpeg_quality_min=90, jpeg_quality_max=98,
                    contrast_min=.92, contrast_max=1.00, gamma_min=.97, gamma_max=1.04, fade_min=.00, fade_max=.06,
                    vignette_min=.00, vignette_max=.06, warm_shift_prob=.08),
        "medium": dict(blur_min=.20, blur_max=.75, downscale_min=1.15, downscale_max=1.75, jpeg_quality_min=76, jpeg_quality_max=92,
                       contrast_min=.82, contrast_max=.96, gamma_min=.92, gamma_max=1.10, fade_min=.04, fade_max=.12,
                       vignette_min=.03, vignette_max=.13, warm_shift_prob=.18),
        "high": dict(blur_min=.60, blur_max=1.35, downscale_min=1.55, downscale_max=2.6, jpeg_quality_min=55, jpeg_quality_max=80,
                     contrast_min=.68, contrast_max=.88, gamma_min=.86, gamma_max=1.18, fade_min=.11, fade_max=.26,
                     vignette_min=.08, vignette_max=.23, warm_shift_prob=.32),
    },
}


def stable_seed(*parts, base: int = 20260930) -> int:
    raw = "|".join(str(x) for x in parts).encode("utf-8")
    return base + int(hashlib.sha256(raw).hexdigest()[:8], 16)


def deterministic_crop(img: Image.Image, size: int, seed: int) -> Image.Image:
    """Preserve aspect ratio, upsize only if needed, then take a deterministic square crop."""
    img = img.convert("RGB")
    w, h = img.size
    if min(w, h) < size:
        scale = size / min(w, h)
        w2, h2 = max(size, round(w * scale)), max(size, round(h * scale))
        img = img.resize((w2, h2), Image.Resampling.LANCZOS)
        w, h = img.size
    rng = random.Random(seed)
    left = 0 if w == size else rng.randint(0, w - size)
    top = 0 if h == size else rng.randint(0, h - size)
    return img.crop((left, top, left + size, top + size))


def random_train_crop(img: Image.Image, size: int, rng: random.Random) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if min(w, h) < size:
        scale = size / min(w, h)
        img = img.resize((max(size, round(w*scale)), max(size, round(h*scale))), Image.Resampling.LANCZOS)
        w, h = img.size
    left = 0 if w == size else rng.randint(0, w-size)
    top = 0 if h == size else rng.randint(0, h-size)
    return img.crop((left, top, left+size, top+size))


def _np(img):
    return np.asarray(img.convert("RGB"), dtype=np.float32)


def _blank_mask(size):
    return Image.new("L", size, 0)


def add_noise(img: Image.Image, severity: str, seed: int, cfg=None):
    p = (cfg or DEFAULT_CONFIG)["noise"][severity]
    rng = np.random.default_rng(seed)
    arr = _np(img)

    sigma = float(rng.uniform(p["sigma_min"], p["sigma_max"]))
    # Old-film grain is primarily luminance-correlated; keep chroma noise smaller.
    lum = rng.normal(0.0, sigma, (arr.shape[0], arr.shape[1], 1))
    chroma = rng.normal(0.0, sigma * p["chroma_scale"], arr.shape)
    out = arr + lum + chroma

    # Tiny sparse bright/dark specks. These are fine-grain noise, not the scratch mask.
    prob = float(p["speckle_prob"])
    selector = rng.random((arr.shape[0], arr.shape[1]))
    bright = selector < prob/2
    dark = (selector >= prob/2) & (selector < prob)
    if bright.any():
        out[bright] = np.maximum(out[bright], rng.uniform(220,255,size=(bright.sum(),1)))
    if dark.any():
        out[dark] = np.minimum(out[dark], rng.uniform(0,35,size=(dark.sum(),1)))

    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out), {"sigma": sigma, "speckle_prob": prob, "chroma_scale": p["chroma_scale"]}


def _curve_points(w, h, rng: random.Random):
    """Generate a smooth scratch path with small local drift rather than angular polylines."""
    mode = rng.choices(["vertical", "diagonal", "short"], weights=[0.58, 0.27, 0.15])[0]

    if mode == "vertical":
        n = rng.randint(18, 32)
        y0 = rng.randint(-h // 10, h // 12)
        y1 = rng.randint(h - h // 12, h + h // 10)
        ys = np.linspace(y0, y1, n)
        x = float(rng.randint(0, w - 1))
        drift = rng.uniform(-0.35, 0.35)
        xs = []
        for _ in range(n):
            drift = 0.88 * drift + rng.gauss(0, 0.45)
            x += drift
            xs.append(x)
        xs = np.asarray(xs)
        # light smoothing removes zig-zag while preserving waviness
        if len(xs) >= 5:
            kernel = np.ones(5) / 5.0
            xs = np.convolve(np.pad(xs, (2, 2), mode="edge"), kernel, mode="valid")
        pts = [(int(np.clip(xv, 0, w - 1)), int(np.clip(yv, 0, h - 1))) for xv, yv in zip(xs, ys)]

    elif mode == "diagonal":
        n = rng.randint(14, 26)
        x0 = rng.randint(-w // 8, w // 3)
        x1 = rng.randint(2 * w // 3, w + w // 8)
        y0 = rng.randint(0, h - 1)
        y1 = int(np.clip(y0 + rng.choice([-1, 1]) * rng.uniform(h * .18, h * .62), 0, h - 1))
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
        noise = np.array([rng.gauss(0, h * .006) for _ in range(n)])
        if len(noise) >= 5:
            noise = np.convolve(np.pad(noise, (2, 2), mode="edge"), np.ones(5) / 5.0, mode="valid")
        ys = ys + noise
        pts = [(int(np.clip(xv, 0, w - 1)), int(np.clip(yv, 0, h - 1))) for xv, yv in zip(xs, ys)]

    else:
        # short curved abrasion / partial scratch, not a full-frame geometric line
        n = rng.randint(10, 20)
        x0, y0 = rng.randint(0, w - 1), rng.randint(0, h - 1)
        angle = rng.uniform(-math.pi, math.pi)
        length = rng.uniform(min(w, h) * .08, min(w, h) * .30)
        ts = np.linspace(0, 1, n)
        bend = rng.uniform(-0.20, 0.20)
        xs = x0 + np.cos(angle) * length * ts + np.sin(angle) * bend * length * np.sin(np.pi * ts)
        ys = y0 + np.sin(angle) * length * ts - np.cos(angle) * bend * length * np.sin(np.pi * ts)
        pts = [(int(np.clip(xv, 0, w - 1)), int(np.clip(yv, 0, h - 1))) for xv, yv in zip(xs, ys)]

    clean = [pts[0]]
    for p in pts[1:]:
        if p != clean[-1]:
            clean.append(p)
    return clean if len(clean) >= 2 else [(0, 0), (w - 1, h - 1)]

def _draw_damage_line(mask_draw, overlay_draw, alpha_draw, pts, width, alpha, value):
    mask_draw.line(pts, fill=255, width=width, joint="curve")
    overlay_draw.line(pts, fill=(value,value,value), width=width, joint="curve")
    alpha_draw.line(pts, fill=int(round(alpha*255)), width=width, joint="curve")


def add_scratches(img: Image.Image, severity: str, seed: int, cfg=None):
    p = (cfg or DEFAULT_CONFIG)["scratch"][severity]
    rng = random.Random(seed)
    base = img.convert("RGB")
    w, h = base.size

    mask = _blank_mask(base.size)
    md = ImageDraw.Draw(mask)
    overlay = Image.new("RGB", base.size, (0, 0, 0))
    od = ImageDraw.Draw(overlay)
    alpha_map = Image.new("L", base.size, 0)
    ad = ImageDraw.Draw(alpha_map)

    n_lines = rng.randint(p["count_min"], p["count_max"])
    branch_count = 0
    for _ in range(n_lines):
        pts = _curve_points(w, h, rng)
        width = rng.randint(p["width_min"], p["width_max"])
        alpha = rng.uniform(p["alpha_min"], p["alpha_max"])
        value = rng.randint(218, 252) if rng.random() < .80 else rng.randint(12, 65)
        _draw_damage_line(md, od, ad, pts, width, alpha, value)

        # Real crack branches are short and smooth; avoid large angular networks.
        if rng.random() < p["branch_prob"] and len(pts) >= 8:
            anchor_idx = rng.randint(len(pts) // 4, 3 * len(pts) // 4)
            anchor = pts[anchor_idx]
            n = rng.randint(6, 12)
            angle = rng.uniform(-math.pi, math.pi)
            length = rng.uniform(min(w, h) * .035, min(w, h) * .12)
            ts = np.linspace(0, 1, n)
            bend = rng.uniform(-.18, .18)
            xs = anchor[0] + np.cos(angle) * length * ts + np.sin(angle) * bend * length * np.sin(np.pi * ts)
            ys = anchor[1] + np.sin(angle) * length * ts - np.cos(angle) * bend * length * np.sin(np.pi * ts)
            bpts = [(int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))) for x, y in zip(xs, ys)]
            _draw_damage_line(md, od, ad, bpts, max(1, width - 1), alpha * .72, value)
            branch_count += 1

    # Sparse abrasion/dust marks.
    speck_count = rng.randint(p["speck_count_min"], p["speck_count_max"])
    for _ in range(speck_count):
        rx = rng.choices([1, 2, 3, 4], weights=[.50, .30, .15, .05])[0]
        ry = max(1, int(rx * rng.uniform(.55, 1.5)))
        x, y = rng.randint(0, w - 1), rng.randint(0, h - 1)
        box = (x - rx, y - ry, x + rx, y + ry)
        value = rng.randint(220, 252) if rng.random() < .82 else rng.randint(8, 55)
        a = rng.uniform(p["alpha_min"] * .55, p["alpha_max"] * .9)
        md.ellipse(box, fill=255)
        od.ellipse(box, fill=(value, value, value))
        ad.ellipse(box, fill=int(a * 255))

    # One local scuff cluster at medium/high severity, made of curved partial abrasions.
    for _ in range(int(p["scuff_clusters"])):
        cx, cy = rng.randint(0, w - 1), rng.randint(0, h - 1)
        for _ in range(rng.randint(4, 9)):
            x0 = int(np.clip(cx + rng.gauss(0, 24), 0, w - 1))
            y0 = int(np.clip(cy + rng.gauss(0, 24), 0, h - 1))
            ang = rng.uniform(-math.pi, math.pi)
            length = rng.randint(7, 30)
            n = 7
            ts = np.linspace(0, 1, n)
            xs = x0 + np.cos(ang) * length * ts + rng.uniform(-3, 3) * np.sin(np.pi * ts)
            ys = y0 + np.sin(ang) * length * ts + rng.uniform(-3, 3) * np.sin(np.pi * ts)
            pts = [(int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))) for x, y in zip(xs, ys)]
            width = rng.randint(1, max(1, p["width_max"] - 1))
            val = rng.randint(222, 252)
            a = rng.uniform(.20, min(.65, p["alpha_max"]))
            _draw_damage_line(md, od, ad, pts, width, a, val)

    # A tiny blur on the alpha only softens digital-hard edges while keeping the GT mask binary.
    alpha_map = alpha_map.filter(ImageFilter.GaussianBlur(radius=.30 if severity != "high" else .40))
    out = Image.composite(overlay, base, alpha_map)
    coverage = float((np.asarray(mask) > 0).mean())
    return out, mask, {
        "line_count": n_lines,
        "branch_count": branch_count,
        "speck_count": speck_count,
        "mask_coverage": coverage,
    }

def _coverage(mask: Image.Image) -> float:
    return float((np.asarray(mask,dtype=np.uint8)>0).mean())


def _edge_tear(mask: Image.Image, rng: random.Random, target_depth: int):
    w,h=mask.size
    d=ImageDraw.Draw(mask)
    side=rng.choice(["top","bottom","left","right"])
    if side in ("top","bottom"):
        span=rng.randint(max(30,w//8),max(40,w//2))
        start=rng.randint(0,max(0,w-span))
        pts=[(start,0 if side=="top" else h-1)]
        segments=rng.randint(5,10)
        for i in range(segments+1):
            x=start+round(i*span/segments)
            depth=rng.randint(max(3,target_depth//5),target_depth)
            y=depth if side=="top" else h-1-depth
            pts.append((x,y))
        pts.append((start+span,0 if side=="top" else h-1))
    else:
        span=rng.randint(max(30,h//8),max(40,h//2))
        start=rng.randint(0,max(0,h-span))
        pts=[(0 if side=="left" else w-1,start)]
        segments=rng.randint(5,10)
        for i in range(segments+1):
            y=start+round(i*span/segments)
            depth=rng.randint(max(3,target_depth//5),target_depth)
            x=depth if side=="left" else w-1-depth
            pts.append((x,y))
        pts.append((0 if side=="left" else w-1,start+span))
    d.polygon(pts, fill=255)


def _organic_blob(mask: Image.Image, rng: random.Random, cx: int, cy: int, rx: int, ry: int):
    """Draw an irregular ragged blob with overlapping lobes."""
    d = ImageDraw.Draw(mask)
    n = rng.randint(14, 26)
    pts = []
    phase1 = rng.uniform(0, 2 * math.pi)
    phase2 = rng.uniform(0, 2 * math.pi)
    for i in range(n):
        a = 2 * math.pi * i / n
        jitter = 1.0 + 0.22 * math.sin(3 * a + phase1) + 0.12 * math.sin(7 * a + phase2) + rng.uniform(-.10, .10)
        x = cx + math.cos(a) * rx * jitter
        y = cy + math.sin(a) * ry * jitter
        pts.append((int(np.clip(x, 0, mask.width - 1)), int(np.clip(y, 0, mask.height - 1))))
    d.polygon(pts, fill=255)

    # Small connected lobes make the boundary less "painted" and more like emulsion loss.
    for _ in range(rng.randint(1, 4)):
        ox = int(np.clip(cx + rng.gauss(0, rx * .65), 0, mask.width - 1))
        oy = int(np.clip(cy + rng.gauss(0, ry * .65), 0, mask.height - 1))
        rrx = max(2, int(rx * rng.uniform(.18, .45)))
        rry = max(2, int(ry * rng.uniform(.18, .45)))
        d.ellipse((ox - rrx, oy - rry, ox + rrx, oy + rry), fill=255)


def _paper_texture(size, rng: random.Random, dark: bool = False):
    w, h = size
    np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
    if dark:
        base = rng.randint(18, 48)
        rgb = np.array([base, base, base], dtype=np.float32)
        noise_sigma = rng.uniform(3, 7)
    else:
        base = rng.randint(214, 238)
        # very mild warm paper tone, never pink
        rgb = np.array([base + rng.randint(0, 4), base, max(0, base - rng.randint(2, 8))], dtype=np.float32)
        noise_sigma = rng.uniform(2, 5)
    noise = np_rng.normal(0, noise_sigma, (h, w, 1))
    arr = np.clip(rgb.reshape(1, 1, 3) + noise, 0, 255).astype(np.uint8)
    tex = Image.fromarray(arr, "RGB").filter(ImageFilter.GaussianBlur(radius=.35))
    return tex


def add_missing(img: Image.Image, severity: str, seed: int, cfg=None):
    p = (cfg or DEFAULT_CONFIG)["missing"][severity]
    rng = random.Random(seed)
    base = img.convert("RGB")
    w, h = base.size
    mask = _blank_mask(base.size)

    target = rng.uniform(p["target_coverage_min"], p["target_coverage_max"])
    blobs = 0
    min_dim = min(w, h)

    # Organic local emulsion-loss patches rather than thick freehand "scribbles".
    while _coverage(mask) < target and blobs < 24:
        blobs += 1
        if severity == "low":
            rx = rng.randint(max(4, min_dim // 90), max(8, min_dim // 35))
        elif severity == "medium":
            rx = rng.randint(max(7, min_dim // 55), max(14, min_dim // 20))
        else:
            rx = rng.randint(max(12, min_dim // 35), max(20, min_dim // 12))
        ry = max(4, int(rx * rng.uniform(.55, 1.55)))
        cx, cy = rng.randint(0, w - 1), rng.randint(0, h - 1)
        _organic_blob(mask, rng, cx, cy, rx, ry)

    edge_added = False
    if rng.random() < p["edge_tear_prob"]:
        depth_max = (
            max(12, min_dim // 28) if severity == "low"
            else max(18, min_dim // 18) if severity == "medium"
            else max(28, min_dim // 11)
        )
        _edge_tear(mask, rng, target_depth=rng.randint(max(6, depth_max // 3), depth_max))
        edge_added = True

    # Simulate exposed paper/emulsion backing with subtle texture rather than flat pink/white paint.
    texture = _paper_texture(base.size, rng, dark=(rng.random() < .12))
    # Feather only the visual composite; the stored mask remains binary.
    soft_mask = mask.filter(ImageFilter.GaussianBlur(radius=.55 if severity != "high" else .75))
    out = Image.composite(texture, base, soft_mask)

    return out, mask, {
        "target_coverage": target,
        "actual_coverage": _coverage(mask),
        "blobs": blobs,
        "edge_tear": edge_added,
    }

def _gamma(img: Image.Image, gamma: float) -> Image.Image:
    arr=np.asarray(img.convert("RGB"),dtype=np.float32)/255.0
    arr=np.clip(arr,0,1)**gamma
    return Image.fromarray(np.clip(arr*255,0,255).astype(np.uint8))


def _vignette(img: Image.Image, strength: float) -> Image.Image:
    if strength <= 0:
        return img
    arr=np.asarray(img.convert("RGB"),dtype=np.float32)
    h,w=arr.shape[:2]
    yy,xx=np.mgrid[0:h,0:w]
    cx,cy=(w-1)/2,(h-1)/2
    rr=np.sqrt(((xx-cx)/max(cx,1))**2+((yy-cy)/max(cy,1))**2)
    factor=1.0-strength*np.clip(rr,0,1.4)**1.6
    arr*=factor[...,None]
    return Image.fromarray(np.clip(arr,0,255).astype(np.uint8))


def add_age_quality(img: Image.Image, severity: str, seed: int, cfg=None):
    p=(cfg or DEFAULT_CONFIG)["age_quality"][severity]
    rng=random.Random(seed)
    x=img.convert("RGB")
    w,h=x.size

    blur=rng.uniform(p["blur_min"],p["blur_max"])
    if blur > .02:
        x=x.filter(ImageFilter.GaussianBlur(blur))

    down=rng.uniform(p["downscale_min"],p["downscale_max"])
    if down > 1.01:
        sw,sh=max(16,round(w/down)),max(16,round(h/down))
        x=x.resize((sw,sh),Image.Resampling.BICUBIC).resize((w,h),Image.Resampling.BICUBIC)

    contrast=rng.uniform(p["contrast_min"],p["contrast_max"])
    x=ImageEnhance.Contrast(x).enhance(contrast)

    gamma=rng.uniform(p["gamma_min"],p["gamma_max"])
    x=_gamma(x,gamma)

    fade=rng.uniform(p["fade_min"],p["fade_max"])
    if fade > 0:
        # Fade toward paper-gray, not pure white.
        paper=Image.new("RGB",x.size,(232,229,221))
        x=Image.blend(x,paper,fade)

    warm=False
    if rng.random() < p["warm_shift_prob"]:
        arr=np.asarray(x,dtype=np.float32)
        # Mild warm/sepia drift only; preserve the source's historical tonality.
        arr[...,0]*=rng.uniform(1.01,1.06)
        arr[...,1]*=rng.uniform(.99,1.03)
        arr[...,2]*=rng.uniform(.91,.99)
        x=Image.fromarray(np.clip(arr,0,255).astype(np.uint8))
        warm=True

    vignette=rng.uniform(p["vignette_min"],p["vignette_max"])
    x=_vignette(x,vignette)

    q=rng.randint(p["jpeg_quality_min"],p["jpeg_quality_max"])
    buf=BytesIO()
    x.save(buf,format="JPEG",quality=q,subsampling=0)
    buf.seek(0)
    x=Image.open(buf).convert("RGB").copy()

    return x,{
        "blur_radius":blur,"downscale":down,"contrast":contrast,"gamma":gamma,
        "fade":fade,"vignette":vignette,"warm_shift":warm,"jpeg_quality":q
    }


def degrade(img: Image.Image, suite: str, severity: str, seed: int, cfg=None):
    if suite not in SUITES:
        raise ValueError(f"Unknown suite: {suite}")
    if severity not in SEVERITIES:
        raise ValueError(f"Unknown severity: {severity}")

    scratch_mask=_blank_mask(img.size)
    missing_mask=_blank_mask(img.size)
    params={}

    if suite=="noise":
        out,p=add_noise(img,severity,stable_seed(seed,"noise"),cfg)
        params["noise"]=p
    elif suite=="scratch":
        out,scratch_mask,p=add_scratches(img,severity,stable_seed(seed,"scratch"),cfg)
        params["scratch"]=p
    elif suite=="missing":
        out,missing_mask,p=add_missing(img,severity,stable_seed(seed,"missing"),cfg)
        params["missing"]=p
    elif suite=="age_quality":
        out,p=add_age_quality(img,severity,stable_seed(seed,"age_quality"),cfg)
        params["age_quality"]=p
    elif suite=="combined":
        out,p=add_age_quality(img,severity,stable_seed(seed,"age_quality"),cfg)
        params["age_quality"]=p
        out,p=add_noise(out,severity,stable_seed(seed,"noise"),cfg)
        params["noise"]=p
        out,scratch_mask,p=add_scratches(out,severity,stable_seed(seed,"scratch"),cfg)
        params["scratch"]=p
        out,missing_mask,p=add_missing(out,severity,stable_seed(seed,"missing"),cfg)
        params["missing"]=p

    return {
        "image":out,
        "scratch_mask":scratch_mask,
        "missing_mask":missing_mask,
        "params":params,
    }
