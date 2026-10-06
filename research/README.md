# Data preparation, training and experiments

Use [DATA.md](../DATA.md) for the frozen dataset and [Windows reproduction commands](../docs/REPRODUCIBILITY.md) for evaluation and experiments.

| Item | Purpose |
|---|---|
| [data_preparation](data_preparation/README.md) | Source selection, attribution and labelled benchmark generation |
| [training](training/README.md) | Final Version training source, initialisation weight and original fixed protocol |
| `prepare_training.py` | Reconstruct the unchanged sealed training bundle in a new work directory |
| `evaluate_detector.py` | Evaluate the supplied selected detector |
| `run_experiments.py` | Compare mask sources and denoising placement |
| `compare_portraits.py` | Compare Microsoft baseline and final pipeline, including face effects |
| `fixtures`, `FIXTURES.json` | Hash-pinned auxiliary scratch/noise probes and controls |
| [denoising](denoising/README.md) | FFDNet comparison adapter, weight and licence |

The fixtures are the exact two scratch probes, five clean controls and two noise probes used in development. They are auxiliary scikit-image examples, not an independent historical-photo benchmark. Labels and clean references are used for scoring after predictions. Personal development portraits are not included; supply your own inputs for qualitative comparison.

The original training source and initialisation checkpoint are retained byte-for-byte. Preparation restores the original operator note in an isolated work directory so sealed checks remain valid; the public training README is in English. This preserves the selected checkpoint and training algorithm.

Original measurements are grouped in [docs/results](../docs/results). Fresh runs create separate output directories without overwriting those records or changing the default restoration pipeline.

FFDNet uses the official KAIR gray weight, SHA256 `0b254d45dafc1ed04729b2206e0c09e5cc1e477e1d094e5a876ea45a34e5d84c`, from https://github.com/cszn/KAIR/releases/download/v1.0/ffdnet_gray_clip.pth. It is an experimental comparison component.
