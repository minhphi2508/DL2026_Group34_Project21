from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from pathlib import Path

from PIL import Image,ImageDraw,ImageFont,ImageOps


SUITES=("noise","scratch","missing","age_quality","combined")
SEVS=("low","medium","high")


def font(size):
    for n in ("DejaVuSans.ttf","arial.ttf"):
        try:return ImageFont.truetype(n,size)
        except Exception:pass
    return ImageFont.load_default()


def panel(gt:Path,inp:Path,scratch:Path|None,missing:Path|None,label:str,out:Path):
    f=font(16)
    imgs=[]
    with Image.open(gt) as g, Image.open(inp) as x:
        imgs=[("GT",g.convert("RGB").copy()),("INPUT",x.convert("RGB").copy())]
    if scratch and scratch.exists():
        with Image.open(scratch) as m: imgs.append(("SCRATCH MASK",m.convert("RGB").copy()))
    if missing and missing.exists():
        with Image.open(missing) as m: imgs.append(("MISSING MASK",m.convert("RGB").copy()))
    cell=300;head=42
    can=Image.new("RGB",(cell*len(imgs),cell+head),"white")
    d=ImageDraw.Draw(can)
    for i,(name,im) in enumerate(imgs):
        th=ImageOps.contain(im,(cell,cell),Image.Resampling.LANCZOS)
        can.paste(th,(i*cell+(cell-th.width)//2,head+(cell-th.height)//2))
        d.text((i*cell+6,8),name,fill="black",font=f)
    d.text((6,head-20),label,fill="black",font=font(12))
    out.parent.mkdir(parents=True,exist_ok=True)
    can.save(out,quality=93,subsampling=0)


def sheet(items,out,cols=2,rows=3):
    if not items:return
    out.parent.mkdir(parents=True, exist_ok=True)
    per=cols*rows
    for st in range(0,len(items),per):
        chunk=items[st:st+per]
        thumbs=[]
        for p in chunk:
            with Image.open(p) as im:
                thumbs.append(ImageOps.contain(im.convert("RGB"),(950,520),Image.Resampling.LANCZOS))
        cw,ch=960,530
        can=Image.new("RGB",(cols*cw,rows*ch),"white")
        for i,im in enumerate(thumbs):
            rr,cc=divmod(i,cols)
            can.paste(im,(cc*cw+(cw-im.width)//2,rr*ch+(ch-im.height)//2))
        page=st//per+1
        can.save(out.with_name(f"{out.stem}_{page:02d}.jpg"),quality=92,subsampling=0)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--benchmark-root",type=Path,default=Path("benchmark_v1"))
    ap.add_argument("--output-root",type=Path,default=Path("benchmark_v1_preview"))
    ap.add_argument("--seed",type=int,default=20260930)
    args=ap.parse_args()

    if args.output_root.exists():shutil.rmtree(args.output_root)
    rng=random.Random(args.seed)
    val_gt=sorted((args.benchmark_root/"val"/"gt").glob("*.png"))
    ids=[p.stem for p in val_gt]
    rng.shuffle(ids)

    all_panels=[]
    cursor=0
    for suite in SUITES:
        for sev in SEVS:
            # two different validation examples per suite/severity
            for j in range(2):
                sid=ids[cursor%len(ids)];cursor+=1
                gt=args.benchmark_root/"val"/"gt"/f"{sid}.png"
                base=args.benchmark_root/"val"/suite/sev
                inp=base/"input"/f"{sid}.png"
                sm=(base/"scratch_mask"/f"{sid}.png") if suite in ("scratch","combined") else None
                mm=(base/"missing_mask"/f"{sid}.png") if suite in ("missing","combined") else None
                out=args.output_root/"panels"/f"{suite}_{sev}_{j+1}_{sid}.jpg"
                panel(gt,inp,sm,mm,f"{suite} | {sev} | {sid}",out)
                all_panels.append(out)

    sheet(all_panels,args.output_root/"contact_sheets"/"benchmark_preview.jpg")

    z=args.output_root.with_suffix(".zip")
    if z.exists():z.unlink()
    with zipfile.ZipFile(z,"w",zipfile.ZIP_DEFLATED) as zf:
        for p in args.output_root.rglob("*"):
            if p.is_file():zf.write(p,p.relative_to(args.output_root.parent))

    print("BENCHMARK PREVIEW CREATED")
    print(f"Panels: {len(all_panels)}")
    print(f"ZIP: {z}")


if __name__=="__main__":
    main()
