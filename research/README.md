# Reproducing the project experiments

Use [../docs/REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md) for Windows commands.
The scripts compare fixed mask sources and denoising order with the same Microsoft restoration backend. `compare_portraits.py` compares Microsoft alone with the final pipeline on a supplied image folder and saves outputs before and after face restoration.

The committed fixtures contain the exact two scratch probes, five undamaged controls, and two noise probes used in development. They are auxiliary scikit-image images, not an independent historical-photo benchmark. Reference images and synthetic scratch labels are accessed only by the evaluator after predictions. Personal development portraits are not included; use your own inputs for qualitative comparison.

`denoising/` contains the group's KAIR-compatible FFDNet adapter, its MIT notice, and the official gray weight used in the comparison. SHA256: `0b254d45dafc1ed04729b2206e0c09e5cc1e477e1d094e5a876ea45a34e5d84c`. Weight source: https://github.com/cszn/KAIR/releases/download/v1.0/ffdnet_gray_clip.pth.

Stored measurements in `../docs/research/` describe the original studies. Fresh runs create separate outputs and do not overwrite those records or change the default restoration policy.
