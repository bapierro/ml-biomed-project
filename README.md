# ML in Biomedicine

Pancreas segmentation across MRI acquisitions. We train on T1-weighted MRI and
study transfer to T2-weighted MRI using nnU-Net with self-similarity features.
**SSC4 is the default:** four patch comparisons at radius 1. SSC8 adds radius 2
as an ablation. nnU-Net provides preprocessing, the U-Net, training and validation.

## Setup

Use Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
From a checkout of this repository:

```sh
uv sync --locked --extra cpu --group ssc --group notebooks
uv run --locked --extra cpu biomed-check
```

The `cpu` profile supports Apple MPS. For NVIDIA GPUs on Linux/Windows, replace
`cpu` with `cuda`. Data, weights and generated results stay local and are not in Git.

## Train SSC4

Download the selected T1 train/validation files from the
[pinned CrossPan release](configs/crosspan-t1/source.json), then train:

```sh
uv run --locked --extra cpu --group ssc python scripts/prepare_crosspan.py --download
uv run --locked --extra cpu --group ssc python scripts/train_ssc.py 801
```

This downloads **369 T1-weighted MRI volumes and their segmentation masks**.
Each volume has one 3D MRI NIfTI file and one matching 3D mask NIfTI file
(738 files total); 2D slices are extracted during training, not stored as separate files.
The helper verifies the files and uses native nnU-Net preprocessing.
Budget roughly **3.9 GB for images/masks**, plus
staging copies and preprocessing overhead. Verified local files are reused.
The official split is **323 training / 46 development-validation volumes**, after
excluding one known train/validation image alias. No TEST or T2 files are selected.
Already have the release? Replace `--download` with `/path/to/CrossPan`.

Training defaults to **SSC4, 2D, 100 epochs, seed 1729 and fold 0**, selecting an
available device automatically. Add `--channels 8`
for the SSC8 ablation, `--dry-run` to inspect a command, or `--resume` to continue
an interrupted fit. Files go under `data/nnunet/`; `--workspace` changes that location.
See the [training guide](docs/ssc-training.md) for data layout, outputs and reproduction details.

## Explore the model

```sh
uv run --locked --extra cpu --group ssc --group notebooks jupyter lab notebooks/ssc-domain-generalization.ipynb
```

The [SSC4 notebook](notebooks/ssc-domain-generalization.ipynb) shows the frozen model's
training curves, full source/target results and interactive reference-versus-prediction
views. Its setup downloads only missing demo files (about **314 MB** on first run):
public weights/results from this repository’s release and six MRI/reference pairs
directly from pinned CrossPan. No GitHub login is required. Verified local files
are reused; a complete
bundle works offline. Optional one-scan inference and provenance are inside the notebook.

[Earlier UniverSeg/TotalSegmentator examples](notebooks/data-visualization-and-universeg.ipynb) and the
[environment check](notebooks/environment-check.ipynb) are also retained.

## Development

Shared code lives in `src/ml_biomed_project/`, entry scripts in `scripts/`, and tests
in `tests/`. Experiment history belongs in `research/`; the public training path
does not import it. The training scripts are small adapters to nnU-Net, not another
training framework.

```sh
uv run --locked --extra cpu --group ssc ruff check src scripts tests notebooks
uv run --locked --extra cpu --group ssc pytest
```

Keep dependencies in `pyproject.toml` and `uv.lock` together. Clear notebook outputs
before committing. Use branches and pull requests for changes.
