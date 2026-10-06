# Installation and verification tools

The root Windows launchers call these scripts automatically. From the repository root:

```powershell
.\.venv\Scripts\python.exe tools\doctor.py --device auto
.\.venv\Scripts\python.exe tools\setup_models.py --verify-only
```

- `setup_windows.py`: creates the environment and installs the CPU or CUDA 12.8 profile.
- `setup_models.py`: downloads and checks the required Microsoft/dlib weights, with resume and local-cache support.
- `doctor.py`: checks imports, the selected device and pinned source/model assets.

Use `SETUP_CPU.cmd` or `SETUP_RTX.cmd` for initial installation. Inference stays at `restore.py` in the repository root.
