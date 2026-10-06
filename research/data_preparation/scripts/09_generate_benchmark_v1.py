from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.degradation_v1 import SUITES,SEVERITIES,deterministic_crop,degrade,stable_seed,DEFAULT_CONFIG


IMG_EXTS={".jpg",".jpeg",".png",".tif",".tiff",".webp"}


def images(path):
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def load_cfg(path: Path):
    if not path.exists():
        return DEFAULT_CONFIG,512,20260930
    raw=yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg={k:raw[k] for k in ("noise","scratch","missing","age_quality")}
    return cfg,int(raw["crop_size"]),int(raw["seed"])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset-root",type=Path,default=Path("dataset_v1"))
    ap.add_argument("--output-root",type=Path,default=Path("benchmark_v1"))
    ap.add_argument("--config",type=Path,default=Path("degradation_config_v1.yaml"))
    ap.add_argument("--overwrite",action="store_true")
    args=ap.parse_args()

    if args.output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.output_root} already exists. Refusing to overwrite. Use --overwrite only before benchmark_v1 is frozen.")
        shutil.rmtree(args.output_root)

    cfg,crop_size,base_seed=load_cfg(args.config)
    manifest=[]

    for split in ("val","test"):
        source_files=images(args.dataset_root/"core_old"/split)
        if len(source_files)!=100:
            raise SystemExit(f"{split}: expected 100 core images, found {len(source_files)}")

        gt_dir=args.output_root/split/"gt"
        gt_dir.mkdir(parents=True,exist_ok=True)

        # Freeze one deterministic 512x512 crop per original image.
        gt_map={}
        for src in tqdm(source_files,desc=f"{split}:GT"):
            sample_id=src.stem
            crop_seed=stable_seed(base_seed,split,sample_id,"crop")
            with Image.open(src) as im:
                gt=deterministic_crop(im,crop_size,crop_seed)
            gt_path=gt_dir/f"{sample_id}.png"
            gt.save(gt_path,format="PNG",optimize=True)
            gt_map[sample_id]=(gt,gt_path,src,crop_seed)

        for suite in SUITES:
            for severity in SEVERITIES:
                base=args.output_root/split/suite/severity
                inp_dir=base/"input"
                inp_dir.mkdir(parents=True,exist_ok=True)
                if suite in ("scratch","combined"):
                    (base/"scratch_mask").mkdir(parents=True,exist_ok=True)
                if suite in ("missing","combined"):
                    (base/"missing_mask").mkdir(parents=True,exist_ok=True)

                for sample_id,(gt,gt_path,src,crop_seed) in tqdm(gt_map.items(),desc=f"{split}:{suite}:{severity}"):
                    deg_seed=stable_seed(base_seed,split,sample_id,suite,severity)
                    r=degrade(gt,suite,severity,deg_seed,cfg)
                    inp_path=inp_dir/f"{sample_id}.png"
                    r["image"].save(inp_path,format="PNG",optimize=True)

                    scratch_path=""
                    missing_path=""
                    if suite in ("scratch","combined"):
                        p=base/"scratch_mask"/f"{sample_id}.png"
                        r["scratch_mask"].save(p,format="PNG",optimize=True)
                        scratch_path=p.as_posix()
                    if suite in ("missing","combined"):
                        p=base/"missing_mask"/f"{sample_id}.png"
                        r["missing_mask"].save(p,format="PNG",optimize=True)
                        missing_path=p.as_posix()

                    manifest.append({
                        "id":sample_id,
                        "split":split,
                        "suite":suite,
                        "severity":severity,
                        "source_core_path":src.as_posix(),
                        "crop_seed":crop_seed,
                        "degradation_seed":deg_seed,
                        "input_path":inp_path.as_posix(),
                        "gt_path":gt_path.as_posix(),
                        "scratch_mask_path":scratch_path,
                        "missing_mask_path":missing_path,
                        "params_json":json.dumps(r["params"],sort_keys=True),
                    })

    meta=args.output_root/"metadata"
    meta.mkdir(parents=True,exist_ok=True)
    with (meta/"benchmark_manifest.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(manifest[0].keys()))
        w.writeheader();w.writerows(manifest)

    (meta/"benchmark_spec.json").write_text(json.dumps({
        "version":"benchmark_v1",
        "seed":base_seed,
        "crop_size":crop_size,
        "splits":["val","test"],
        "suites":list(SUITES),
        "severities":list(SEVERITIES),
        "gt_images_per_split":100,
        "degraded_examples":len(manifest),
        "generation_order_for_combined":["age_quality","noise","scratch","missing"],
        "test_policy":"frozen; never use for model selection or generator tuning",
    },indent=2),encoding="utf-8")

    shutil.copy2(args.config,meta/"degradation_config_v1.yaml")
    print(f"\nBENCHMARK V1 GENERATED: {args.output_root}")
    print(f"Degraded examples: {len(manifest)}")
    print("Expected: 2 splits x 5 suites x 3 severities x 100 = 3000")


if __name__=="__main__":
    main()
