# Dataset preparation on Windows

The group prepares a historical-photo core from Finna-HKM and Finna-JOKA, then adds controlled noise, scratches, missing regions, quality degradation, and combined damage. SynOld and Microsoft's example photographs are separate external sources. See [../../DATA.md](../../DATA.md) for exact versions, frozen hashes, attribution, and download information.

## Reuse the frozen processed package

This is the recommended route for reproducing the report. Extract the complete package into the repository's `data` directory. Keep its split and metadata unchanged. The supplied source selection reflects manual review and recorded training-side replacements; repeating a fresh review does not reproduce the same dataset.

## Preparation source

The scripts and shared modules are included in this directory:

| Script | Purpose |
|---|---|
| `01_fetch_finna_candidates.py` | Download deterministic source candidates and attribution |
| `02_make_contact_sheets.py` | Generate review sheets |
| `03_finalize_finna.py` | Freeze reviewed selection and source-level splits |
| `08c_apply_core_reaudit.py` | Record the original training-side replacements |
| `09_generate_benchmark_v1.py` | Generate paired benchmark conditions from the frozen configuration |
| `10_audit_benchmark_v1.py` | Check counts, shapes, masks, and severity ordering |
| `04_fetch_synold.py` | Fetch and split separate paired scratch examples |
| `05_fetch_microsoft_real.py` | Fetch separate unpaired qualitative examples |

Do not change the 600/100/100 source split after generating variants. Real photographs without clean counterparts are qualitative data; do not report paired PSNR/SSIM for them. SynOld's left image is the damaged input and its right image is the reference. Per-source rights and attribution are retained in `LICENSE_NOTES.md` and metadata.

## Regenerate the controlled benchmark in a new directory

After installing the project environment, from the repository root:

```powershell
New-Item -ItemType Directory -Path work\benchmark_rebuild
Copy-Item -Recurse data\dataset_v1 work\benchmark_rebuild\dataset_v1
Copy-Item data\benchmark_v1_candidate2\metadata\degradation_config_v1.yaml work\benchmark_rebuild\degradation_config_v1.yaml
Set-Location work\benchmark_rebuild
..\..\.venv\Scripts\python.exe ..\..\research_archive\data_builder\scripts\09_generate_benchmark_v1.py --dataset-root dataset_v1 --output-root benchmark_v1_candidate2 --config degradation_config_v1.yaml
..\..\.venv\Scripts\python.exe ..\..\research_archive\data_builder\scripts\10_audit_benchmark_v1.py --benchmark-root benchmark_v1_candidate2
Set-Location ..\..
```

The generator processes validation and test. This command is for post-selection reproduction in an isolated directory, not for choosing or tuning the pipeline. It does not overwrite the downloaded data. Compare the resulting benchmark manifest and validation asset hashes with the frozen values in DATA.md and the detector protocol before calling it the same benchmark.
