# Group data preparation and training source

This directory contains the original data builder and the source used to train the **Final Version** damage detector. Inference setup does not train a model or download the dataset.

- [data_builder/README.md](data_builder/README.md): data preparation on Windows.
- [training/v3/README_VI.md](training/v3/README_VI.md): English training guide. The historical filename is retained for provenance.
- [../DATA.md](../DATA.md): dataset sources, frozen versions, splits, download information, and hashes.
- [../docs/REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md): evaluation and pipeline experiment commands.

The selected model is supplied in `models/v3/`. Its archived protocol retains the original internal identifiers. The training initialisation checkpoint is included. `research/prepare_training.py` reconstructs the exact sealed source bundle in a new work directory, including its original operator note, so that the original hash checks remain valid while all tracked README files are in English. This does not alter the selected checkpoint or training algorithm.
