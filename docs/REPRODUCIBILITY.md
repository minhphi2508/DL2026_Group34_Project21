# Reproducing the main results on Windows

Run commands from the repository root after `SETUP_CPU.cmd` or, for **NVIDIA RTX 5060 Ti 16 GB**, `SETUP_RTX.cmd`. The CPU profile was used for published pipeline comparisons. New results go into fresh directories under `work`; they do not change the fixed inference pipeline or overwrite original experiment records.

## 1. Verify and evaluate the selected detector

Download and extract the frozen processed package as described in [../DATA.md](../DATA.md).

```powershell
.\.venv\Scripts\python.exe research\evaluate_detector.py --data-root data --verify-only
.\.venv\Scripts\python.exe research\evaluate_detector.py --data-root data --output work\detector_validation --device cuda
```

Use `--device cpu` if no GPU is available; a full 1,600-condition pass can take substantially longer. The evaluator verifies 600 training sources and 2,800 validation assets, evaluates the supplied epoch-14 checkpoint with the original thresholds, and saves per-case CSV, summary JSON, and masks. It does not open test pixels.

Expected reported values, rounded to four decimals: **scratch Dice 0.7437**, **missing Dice 0.9513**, **missing recall 0.9455**. Each Dice is a macro average over its 600 positive conditions. These are segmentation metrics rather than final restoration scores. The original numerical gate status remains false; academic selection did not redefine the gate. Differences across devices should be examined using per-case data, not concealed as exact numerical identity.

## 2. Compare damage mask sources

```powershell
.\.venv\Scripts\python.exe research\run_experiments.py --study mask_sources --verify-only
.\.venv\Scripts\python.exe research\run_experiments.py --study mask_sources --output work\mask_sources --device cpu
```

The fixed auxiliary subset includes the two published scratch probes and five undamaged controls. It supplies exact original processed inputs and cached Microsoft scratch masks, verified by hashes, so geometry and scratch detection are held fixed. The group heads are inferred again; only the source of the repair mask changes:

- A: Microsoft scratch + Final Version missing (the final configuration).
- B: Final Version scratch + missing.
- C: Microsoft scratch + both Final Version heads.

Reference images and synthetic labels are used only for scoring, never as inference inputs. `RESULTS.csv` and `RESULTS.json` include mask counts, recall, and global before-face scores. `--masks-only` skips restoration; `--case scratch_only_camera` restricts the run to that one case.

| Probe | A recall (%) / PSNR | B recall (%) / PSNR | C recall (%) / PSNR |
|---|---|---|---|
| Astronaut | 72.12 / 22.082 | 36.29 / 21.274 | 75.50 / 22.183 |
| Camera | 53.85 / 21.149 | 28.01 / 20.207 | 59.99 / 21.667 |

The extra union marks 301 additional pixels across four of five controls in the original mask comparison. Increasing recall on two probes does not establish improved quality on all real photographs. This portable subset reproduces the published auxiliary measurements, not all 22 original conditions or nine real/probe restoration cases; original study records remain in [research/detector_fusion](research/detector_fusion).

## 3. Compare denoising placement

```powershell
.\.venv\Scripts\python.exe research\run_experiments.py --study denoising_order --verify-only
.\.venv\Scripts\python.exe research\run_experiments.py --study denoising_order --output work\denoising_order --device cpu
```

The exact two artificial noise inputs and prespecified empty masks are included. FFDNet uses the original gray weight, automatic noise estimation, blend 0.75, and fixed settings. Compare baseline global restoration, denoising before restoration, denoising after the same baseline output, and FFDNet alone. No detector is rerun after denoising. Scores are measured before face restoration, as in the report.

| Probe | Baseline | Denoise before | Denoise after | FFDNet alone |
|---|---:|---:|---:|---:|
| Astronaut PSNR | 27.20 | 27.26 | 27.30 | 33.97 |
| Camera PSNR | 25.37 | 24.18 | 25.44 | 32.76 |

These are development probes, not a new independent benchmark. Stored six-condition order results, including the original mixed-damage cases, are in [research/pipeline_order](research/pipeline_order). Extra denoising is excluded from the default restoration pipeline.

## 4. Baseline versus final pipeline and face effects

Put images into a folder, then run:

```powershell
.\.venv\Scripts\python.exe research\compare_portraits.py --input inputs --output work\portrait_comparison --device cpu
```

The final branch runs the unchanged default pipeline. The baseline reuses exactly the same Microsoft processing images and scratch masks, omitting only the additional missing-region mask. It runs the same global and face models. Inspect `microsoft_baseline/final`, `final_pipeline/global/restored_image`, and `final_pipeline/final`. `COMPARISON.json` records added mask pixels without claiming those pixels are correctly repaired.

The five original personal portraits are not redistributed in Git. Using other inputs reproduces the comparison procedure but not those exact illustrations. Real images without clean references are evaluated visually, with attention to damage removal, eye/mouth details, object shape, colour, and blending. Do not compute quality PSNR/SSIM against the damaged input.

## 5. Reproduce training

The training source, initialisation checkpoint, loss, fixed schedule, and hash-bound protocol are included. Follow [the English training guide](../research_archive/training/v3/README_VI.md) to prepare a new sealed work directory and run preflight, CUDA smoke, and training on **NVIDIA RTX 5060 Ti 16 GB**. The selected supplied checkpoint allows evaluation without retraining. Exact weight bytes are not guaranteed across hardware/runtime changes.

## Recorded checks

Model/source identity is checked by `doctor.py`; archived implementation parity and clone checks are in [VALIDATION.md](VALIDATION.md). Fresh reproduction runs have their own protocol and status files. A completed command establishes execution, while quality conclusions require the reported scores and visual comparisons.
