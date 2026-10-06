# Training the Final Version damage detector

The historical directory and filename identify the original experiment. The selected detector is a two-head U-Net with a ResNet34 encoder; the final restoration pipeline uses its missing-region head. Scratch and missing thresholds are fixed at 0.4 and 0.5.

## Windows with NVIDIA RTX 5060 Ti 16 GB

Set up the repository with `SETUP_RTX.cmd`, then extract the frozen processed dataset described in [../../../DATA.md](../../../DATA.md) into the repository's `data` directory. It must contain `dataset_v1` and `benchmark_v1_candidate2`.

From the repository root, run in PowerShell:

```powershell
.\.venv\Scripts\python.exe research\prepare_training.py --output work\final_training
.\.venv\Scripts\python.exe work\final_training\runner.py preflight --data-root data --run-name final_reproduction
.\.venv\Scripts\python.exe work\final_training\runner.py smoke --data-root data --run-name final_reproduction --device cuda
.\.venv\Scripts\python.exe work\final_training\runner.py train --data-root data --run-name final_reproduction --device cuda
```

Preparation verifies the included initialisation checkpoint and reconstructs the original sealed source bundle in a new directory. Training uses 600 source photographs, a fixed paired sampling schedule, 15 maximum epochs, batch size 4, AdamW, and validation-based selection. The protocol and source files specify the full configuration. CUDA 12.8 and the exact PyTorch profile are required for this archived training run.

Training saves completed epochs, validation summaries, selected masks, and resume state under `work/final_training/runs/final_reproduction`. Resume only a completed epoch, with the same arguments plus `--resume`. Use a new run name for a fresh experiment. Hardware and library differences can affect exact checkpoint bytes.

For evaluating the supplied selected checkpoint, use `research/evaluate_detector.py` as documented in [../../../docs/REPRODUCIBILITY.md](../../../docs/REPRODUCIBILITY.md). The original runner's `evaluate` action requires a numerically eligible checkpoint and is not the appropriate command for the academically reviewed selected checkpoint. Evaluation records numerical gate status honestly; it does not redefine selection criteria.
