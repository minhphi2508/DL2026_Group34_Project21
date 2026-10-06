# Dataset sources and exact reproduction data

## Downloads

| Package | Link / status | Purpose |
|---|---|---|
| Original source snapshot, `dataset.zip` (668,521,112 bytes) | [Google Drive source snapshot](https://drive.google.com/file/d/1quvHNiHL6aVc8Zh1o07EtT-R6wjAWQmt/view?usp=drive_link) | Source images and original metadata; this archive does not contain the processed benchmark |
| Frozen processed core and benchmark, `ASTRA_PROCESSED_FINNA_CORE_BENCHMARK_20261006.zip` | [Google Drive processed dataset folder](https://drive.google.com/drive/folders/1SGHZ--JN_dDSYwyZ4NW6qRhSqc6sDKTK?usp=drive_link) | Exact data used for detector training/evaluation and controlled benchmark reproduction |
| Auxiliary pipeline experiment fixtures | Included in `research/fixtures`, indexed by `research/FIXTURES.json` | Exact fixed scratch/noise probes and undamaged controls from the pipeline experiments |

The processed ZIP is **1,398,181,622 bytes**, contains **6,411 files**, and has SHA256:

```text
6c49a301ccc37c7513d26b94a707ec422b694b119191e2549789c1290d223029
```

Download the processed package into the repository root. In Windows PowerShell:

```powershell
Get-FileHash .\ASTRA_PROCESSED_FINNA_CORE_BENCHMARK_20261006.zip -Algorithm SHA256
Expand-Archive .\ASTRA_PROCESSED_FINNA_CORE_BENCHMARK_20261006.zip -DestinationPath .\data
```

Check that the hash matches before extracting. Use a new `data` directory. The required resulting paths are `data/dataset_v1` and `data/benchmark_v1_candidate2`. The original source snapshot is not a substitute for the frozen processed archive. Dataset downloads are not needed to restore a user's photograph.

## Official sources and roles

| Source | Official URL | Role |
|---|---|---|
| Finna-HKM | https://huggingface.co/datasets/NatLibFi/Finna-HKM-images | Historical photographs from Helsinki City Museum; core source |
| Finna-JOKA | https://huggingface.co/datasets/NatLibFi/Finna-JOKA-images | Historical photographs from Journalistic Picture Archive JOKA; core source |
| SynOld | https://github.com/wushunshun/SynOld | Separate paired scratch examples; excluded from the 800-image Finna core |
| Microsoft | https://github.com/microsoft/bringing-old-photos-back-to-life | Separate real qualitative examples and restoration implementation; no matching clean reference |

Finna dataset cards specify CC-BY-4.0. Preserve per-image photographer, title, record ID, institution, source URL, and rights in `dataset_v1/metadata/attribution.csv`. Separate external sources retain their own terms; see `research_archive/data_builder/LICENSE_NOTES.md`. The processed download is the Finna core and benchmark package, not a claim that all external sources share one redistribution licence.

## Version and split

The project version is **dataset_v1**; the fixed benchmark directory is **benchmark_v1_candidate2**, prepared in September 2026. Source split seed: **20260929**. The original downloader did not pin an upstream dataset release, so exact identity is defined by the frozen selection, file hashes, and processed archive rather than by today's upstream main branch.

| Source | Training | Validation | Test | Total |
|---|---:|---:|---:|---:|
| HKM | 300 | 50 | 50 | 400 |
| JOKA | 300 | 50 | 50 | 400 |
| Total | 600 | 100 | 100 | 800 |

Manual review separates sufficiently intact historical photographs, naturally damaged photographs, and rejects before splitting. Variants of one source never cross splits. Recorded training-side re-audit replacements are included in the frozen metadata. Repeating subjective review would create a different dataset.

Frozen metadata SHA256:

| File | SHA256 |
|---|---|
| `dataset_v1/metadata/core_manifest.csv` | `abb4fb1c6bf6e0de3cb26d459021be75c3b05614d5e896827ca4c5b6ab6aab1b` |
| `benchmark_v1_candidate2/metadata/benchmark_manifest.csv` | `e560b5c86bf6126a76cfd068a0639bccfbcaa382df37b7773f523e2f7ae3c7cd` |

The processed package includes split lists, per-image attribution, original core images, generated benchmark images, masks, and the frozen degradation configuration. Separate naturally damaged, SynOld, and Microsoft collections are described by their original builder manifests; they are not part of the processed Finna benchmark download.

## Preprocessing and labels

Original core images are retained. The benchmark uses one deterministic 512 × 512 crop per source, reused across five tracks—noise, scratch, missing, age_quality, and combined—at low, medium, and high severity. Per-source crop and degradation seeds are recorded in the benchmark manifest. Synthetic scratch and missing masks label injected damage; original scans can contain unlabelled natural defects.

Validation has 1,500 damaged conditions from 100 sources, plus 100 clean controls for the detector audit. These are correlated variants, not 1,600 independent photographs. Each positive scratch or missing group contains 600 conditions. Training uses the 600 training sources, shared dihedral augmentation, and the fixed paired sampling schedule in the Final Version protocol. ImageNet normalisation is a model input transform, not a modification to stored RGB references.

## Preparation scripts and reproducibility

All preparation scripts are included in `research_archive/data_builder/scripts`, with shared code in `research_archive/data_builder/src`. The [builder README](research_archive/data_builder/README.md) describes their roles and gives Windows commands to regenerate the benchmark in an isolated directory from the frozen selection/configuration.

The source acquisition/review scripts explain the preparation process. Exact report reproduction should use the supplied processed archive and compare its hashes; fresh source sampling or review is not an exact substitute. Do not overwrite the downloaded benchmark or use test results to tune the selected pipeline.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for selected-detector evaluation, training, mask source comparisons, denoising order, and qualitative baseline comparisons. Real photographs without clean references are assessed visually; no paired PSNR/SSIM is assigned to them.
