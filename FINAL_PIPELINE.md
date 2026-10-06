# Selected final pipeline

The default command is `python restore.py --input <image-or-folder>`.

RGB decode → Microsoft input at maximum side 512 → Microsoft scratch >=0.4 OR Final Version native missing >=0.5 with radius-3 elliptical margin → Microsoft scratch+quality global restoration → dlib 68 landmarks / FaceSR256 / official warp-blend when faces are found → PNG.

Only Final Version missing contributes to the final mask. Final Version scratch, C3, MAT, FFDNet and Restormer are not in this selected recipe. FP32/TF32 off, HR off, Torch threads 2. Native-display outputs use LANCZOS resizing, not super-resolution.

See [detailed selection](docs/FINAL_PIPELINE.md), [validation](docs/VALIDATION.md), [models](provenance/MODELS.json) and [research](docs/RESEARCH.md). Source and model hashes are verified; a successful run is not a claim of restoration quality. No new held-out benchmark score is claimed.
