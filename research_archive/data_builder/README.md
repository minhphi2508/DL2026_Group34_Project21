# OLD / DAMAGED Photo Restoration Dataset v1 — Builder

This package builds the shared dataset for the 5-person Deep Learning final project.

## Locked source set

We intentionally use exactly four public sources:

1. **NatLibFi / Finna-HKM-images**
   - historical photographs from Helsinki City Museum
   - 5,947 images, until 1917
   - CC BY 4.0
   - role: **core old-photo source + natural-damage holdout**

2. **NatLibFi / Finna-JOKA-images**
   - historical photographs from Journalistic Picture Archive JOKA
   - 4,595 images, until 1940
   - CC BY 4.0
   - role: **core old-photo source + natural-damage holdout**

3. **wushunshun / SynOld**
   - 500 training pairs + 200 test pairs
   - synthetic scratch restoration pairs
   - each JPEG contains **damaged image on the LEFT, clean GT on the RIGHT**
   - role: **auxiliary scratch data + frozen external synthetic test**
   - repository does not state a license in its README/repository metadata; keep it internal to the class project unless permission/terms are clarified.

4. **Microsoft / Bringing-Old-Photos-Back-to-Life**
   - official research repository
   - `test_images/old`: 8 real old photos
   - `test_images/old_w_scratch`: 4 real scratched old photos
   - role: **frozen external real qualitative test only**
   - no clean GT, so do not compute paired PSNR/SSIM against these images.

## Dataset design

The quantitative core is deliberately **OLD first, DAMAGED second**:

```text
real historical photo from Finna
        ↓
manual review → keep only "clean-ish old" photos as pseudo-ground-truth
        ↓
controlled synthetic degradation created later
        ↓
paired (damaged old image, clean-ish old GT)
```

Do not use modern pristine images as the primary GT domain.

A second real-world branch contains genuinely damaged old photos with no GT:

```text
Finna natural damage + Microsoft real old photos
        ↓
qualitative / no-reference evaluation
```

SynOld is separate because it is a useful paired scratch benchmark, but its underlying photos are not guaranteed to be genuinely historical.

---

# Phase 1 — Build the OLD core now

## Target counts

From Finna:
- HKM: review 600 candidates → keep exactly 400 core images
- JOKA: review 600 candidates → keep exactly 400 core images

Final core:
- train: 600 = 300 HKM + 300 JOKA
- val: 100 = 50 HKM + 50 JOKA
- test: 100 = 50 HKM + 50 JOKA

The split is performed **after manual selection and before any synthetic degradation**.

This prevents the same original historical photograph from leaking across train/val/test.

## Step 0 — environment

Use one project-level environment. Do not create separate requirements per team member.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Step 1 — download deterministic Finna candidate pools

```bash
python scripts/01_fetch_finna_candidates.py \
  --work-root work \
  --per-source 600 \
  --seed 20260929
```

The script:
- downloads only `metadata.jsonl` first;
- filters for CC BY 4.0 records with an image;
- samples candidates deterministically;
- downloads images one by one until 600 technically usable files are obtained per source;
- keeps original-resolution source files;
- writes a manifest and a review CSV.

It does **not** resize/crop/normalize the originals.

## Step 2 — create contact sheets

```bash
python scripts/02_make_contact_sheets.py \
  --work-root work
```

This creates contact sheets under:

```text
work/contact_sheets/HKM/
work/contact_sheets/JOKA/
```

Each image is labeled by its local filename/year.

## Step 3 — manually review candidates

Edit:

```text
work/review/HKM_review.csv
work/review/JOKA_review.csv
```

Set `decision` to exactly one of:

- `core` — old historical photo that is clean-ish enough to serve as GT
- `real_damage` — genuinely old and visibly damaged; save for qualitative testing
- `reject` — unsuitable

### What counts as `core`

Keep:
- genuinely historical photographic appearance;
- natural grain, modest fading, moderate blur, sepia/B&W are OK;
- enough meaningful visual content;
- no large missing region;
- no dominant tear;
- no severe scratch network;
- no major scan failure or watermark;
- not a map, text page, document scan, poster, or mostly blank frame.

The GT does **not** need to look modern or perfectly restored. It only needs to be sufficiently intact that adding controlled damage remains meaningful.

### What counts as `real_damage`

Keep separately when the image is genuinely old and has:
- real scratches;
- tears;
- missing emulsion / missing regions;
- severe age-related degradation;
- strong stains/damage that make it a useful real-world stress case.

These images have no clean GT, so they are never used for paired PSNR/SSIM evaluation.

### Core quota

Mark **exactly 400 HKM** and **exactly 400 JOKA** as `core`.

If there are not enough suitable candidates, fetch a second candidate round with a different seed instead of lowering the standard.

## Step 4 — freeze and split the Finna core

```bash
python scripts/03_finalize_finna.py \
  --work-root work \
  --dataset-root dataset_v1 \
  --seed 20260929
```

Output:

```text
dataset_v1/
├── core_old/
│   ├── train/
│   │   ├── HKM/
│   │   └── JOKA/
│   ├── val/
│   │   ├── HKM/
│   │   └── JOKA/
│   └── test/
│       ├── HKM/
│       └── JOKA/
├── external_real/
│   └── finna_damaged/
│       ├── HKM/
│       └── JOKA/
└── metadata/
    ├── core_manifest.csv
    ├── real_damage_manifest.csv
    └── attribution.csv
```

After this succeeds, the 800-image base is **frozen**.

Do not reshuffle the source split after synthetic degradation is created.

---

# Phase 2 — Add the two external sources

## SynOld

```bash
python scripts/04_fetch_synold.py \
  --dataset-root dataset_v1
```

Output:

```text
dataset_v1/external_synthetic/synold/
├── train/
│   ├── input/
│   └── gt/
└── test/
    ├── input/
    └── gt/
```

The script splits each source JPEG vertically at the midpoint:
- left half = scratched/degraded input
- right half = clean GT

`test/` must remain an external frozen benchmark.

## Microsoft real test images

```bash
python scripts/05_fetch_microsoft_real.py \
  --dataset-root dataset_v1
```

Output:

```text
dataset_v1/external_real/microsoft/
├── old/            # expected 8
└── old_w_scratch/  # expected 4
```

These images are qualitative only.

---

# Phase 3 — Audit before degradation generation

```bash
python scripts/06_audit_sources.py \
  --dataset-root dataset_v1
```

Expected critical checks:
- core total = 800;
- train/val/test = 600/100/100;
- HKM and JOKA each contribute 300/50/50;
- no exact file hash appears in more than one core split;
- SynOld expected 500 train pairs + 200 test pairs;
- SynOld input and GT dimensions match;
- Microsoft expected 8 + 4 images.

Only after this audit passes should the leader build `paired_damage/`.

---

# Why degradation generation is deliberately NOT frozen in this package yet

Before generating synthetic scratches/noise/missing regions, inspect the final 800 old-photo GT images and the naturally damaged holdout.

The synthetic degradation generator should be calibrated against the real damage actually present in the selected historical photos.

That is the next checkpoint.

Do **not** let individual team members generate different versions of the dataset.
