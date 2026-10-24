# ML in Biomedicine

Aalto course project on segmentation in medical imaging. This repository contains the
Python environment, notebooks, and scripts for trying segmentation models. The dataset,
imaging modality, and models are still being chosen.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.13.5 or newer.
You need access to the private GitHub repository. From a terminal:

```sh
git clone https://github.com/bapierro/ml-biomed-project.git
cd ml-biomed-project
uv sync --locked --extra cpu --group notebooks
uv run --locked --extra cpu biomed-check
uv run --locked --extra cpu --group notebooks jupyter lab
```

uv manages Python 3.12 and the project environment; no manual activation is needed.
`biomed-check` reports the environment and tests a small calculation on the selected device.

Use `cpu` for Apple silicon Macs or CPU-only Linux/Windows. This profile includes MPS
acceleration on Apple silicon. For NVIDIA GPUs on Linux/Windows, replace `cpu` with `cuda`
in each command. That profile uses CUDA 12.8 and requires a
[compatible NVIDIA driver](https://docs.nvidia.com/deploy/cuda-compatibility/).
CUDA packages take several GB; a separate CUDA toolkit is unnecessary.

Supported platforms: Apple silicon with macOS 14+, and x86-64 Linux (glibc 2.28+) or Windows.

## Notebooks

- [Environment check](notebooks/environment-check.ipynb): verify packages and device access.
- [Data visualization and UniverSeg](notebooks/data-visualization-and-universeg.ipynb):
  inspect the current MRI examples and compare saved segmentations. It explains support
  sets, preprocessing, and Dice scores. It does not download files or run models.

## Current MRI example

The current example uses [CrossPan](https://huggingface.co/datasets/linkai-peng/CrossPan).
Download its ten selected MRI/mask pairs (**38.4 MiB**):

```sh
uv run --locked --extra cpu python scripts/download_mri_examples.py
```

The [scan inventory](data/crosspan_examples.csv) lists the selected files and patient
identity evidence. The dataset revision is pinned in the downloader. Existing downloads
are reused. With these files alone, the visualization notebook can show images and masks.

To generate UniverSeg predictions for the notebook:

```sh
uv sync --locked --extra cpu --group models --group notebooks
uv run --locked --extra cpu python scripts/download_mri_examples.py --with-universeg-weights
uv run --locked --extra cpu --group models python scripts/run_mri_examples.py --model universeg
```

The checkpoint adds **4.5 MiB**. For TotalSegmentator, use `--model totalsegmentator` instead;
its first run downloads about **442 MiB** of weights, with additional space for extraction.
The runner selects the available device and reuses saved outputs. Add `--dry-run` to inspect
the commands without inference.

These are development examples. Query and support choices are set in
[scripts/run_mri_examples.py](scripts/run_mri_examples.py), where patient separation is
checked for UniverSeg. The inventory assigns no experiment roles. CrossPan does not provide
a complete T1/T2 patient crosswalk, so the examples are not a verified paired cohort.
Data and model use must follow the [CrossPan license](https://huggingface.co/datasets/linkai-peng/CrossPan)
and [UniverSeg model license](https://github.com/JJGO/UniverSeg/blob/833a0c34c65e38d675e21bd48ddec6797cc03259/LICENSE-model).

## Development

Shared code is in `src/ml_biomed_project/`, runnable examples in `scripts/`, and tests in
`tests/`. [Initial literature notes](docs/research-notes.md) collect the papers considered.

Add packages with `uv add --no-sync PACKAGE` or `uv add --dev --no-sync PACKAGE`, then repeat
`uv sync` with your selected profile and groups. Commit `pyproject.toml` and `uv.lock` together.

```sh
uv run --locked --extra cpu ruff check .
uv run --locked --extra cpu ruff format --check .
uv run --locked --extra cpu pytest
```

Use branches and pull requests for changes. Keep scans, weights, and generated results
local in `data/`, `models/`, and `results/`; these artifacts are ignored by Git.
Clear notebook outputs before committing. Update this README when commands or file paths change.
