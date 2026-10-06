# Verification and scope

The completed integration was verified on Windows CPU, Python 3.12.14, PyTorch 2.8.0+cpu and torchvision 0.23.0+cpu. The GPU installation profile targets **NVIDIA RTX 5060 Ti 16 GB**, CUDA 12.8 wheels; full GPU restoration parity with CPU has not been established.

The checks include:

- Five authorised development images in JPEG/JFIF, WebP and PNG: processing inputs, combined masks, global outputs and final outputs match the previously selected implementation pixel-for-pixel. Three cases include face refinement; two retain global output.
- A separate CPU environment and an earlier GitHub clone completed the full portrait pipeline with identical final pixels.
- Four input checks cover supported formats, Unicode names, same-stem files, invalid images and non-recursive folder scanning. GitHub Actions runs these input checks only.
- `tools/doctor.py` checks imports, a real device computation, 76 pinned upstream source files and seven model assets.
- The submission reproduction checks match the recorded mask/noise measurements and both portrait comparison branches. The complete original training bundle can be reconstructed without changing its protocol, source or checkpoint bytes.

Machine-readable records are grouped in [evidence](evidence). The environment listing and historical clone reports describe the runs recorded on their dates; they are not a new test of every later revision.

`evidence/PIPELINE_FROZEN_BEFORE_TEST.yaml` preserves the configuration and file hashes captured on 2026-10-06. Those code hashes describe that historical snapshot, not the current reorganised file layout. Current installation verifies the active identities through `provenance/MODELS.json`, `UPSTREAM_SOURCE.json` and the detector descriptor.

No new full training run or complete 1,600-condition detector validation pass was performed while packaging. Benchmark test pixels were not opened for packaging or tuning. Personal photographs are not committed to Git. Execution parity verifies integration; it does not guarantee damage removal, historical accuracy or faithful face details on every input.
