# ML in Biomedicine project

Medical image segmentation experiments for a three-person Aalto course project.
The final dataset, anatomy, and model comparison await supervisor feedback.
Pancreas segmentation across MRI sequences is a **proposed direction**, not a settled scope.
See [research notes](docs/research-notes.md) for the discussion and references.

This repository provides the environment, device checks, and MRI visualization notebooks.
It contains no datasets, model weights, training pipeline, or medical experiment results.

## First setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) **0.13.5 or newer**.
On macOS with Homebrew, use `brew install uv` (or `brew upgrade uv` if already installed).
On Windows, use `winget install --id=astral-sh.uv -e`; reopen your terminal afterward.
On Linux, follow uv's official installation instructions.

Clone the private repository using your GitHub access, then enter it:

```sh
git clone https://github.com/bapierro/ml-biomed-project.git
cd ml-biomed-project
```

Choose **one** PyTorch profile. Include the same profile in subsequent `uv run` and `uv sync`
commands. Switching profiles lets uv replace the PyTorch build; do not mix them or use `--all-extras`.

### Apple silicon Mac, or CPU-only Linux/Windows

```sh
uv sync --locked --extra cpu --group notebooks
uv run --locked --extra cpu biomed-check
```

The `cpu` profile includes **MPS acceleration on Apple silicon**. Its name refers to the
official wheel index; the Mac wheel supports both CPU and MPS. The check chooses MPS when available.

### Linux/Windows with an NVIDIA GPU

```sh
uv sync --locked --extra cuda --group notebooks
uv run --locked --extra cuda biomed-check --device cuda
```

The CUDA profile uses PyTorch **2.11.0 + CUDA 12.8** from the official PyTorch index.
The CPU/MPS profile uses the same PyTorch release. These builds have matching Python 3.12 wheels
for Apple silicon macOS and 64-bit x86 Linux/Windows.

Install a compatible NVIDIA driver before using CUDA; check `nvidia-smi` and your GPU model.
CUDA 12.x minor compatibility requires at least Linux driver **525.60.13** or Windows
**528.33**, with limitations; a current driver supporting CUDA 12.8 (Linux **570.26** /
Windows **570.65** or newer) is recommended. GPU architecture must also be supported by the
PyTorch wheel. See [NVIDIA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)
and [CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html).
The Python packages provide the CUDA runtime; a separate CUDA toolkit is unnecessary for these
checks. This setup does not install drivers or a toolkit.

CUDA packages can require several GB of downloads and substantially more installed space.
On a CPU-only machine, use the `cpu` profile to avoid that download. Exact sizes depend on the
platform and cache.

Python is pinned to 3.12. uv reuses a compatible installed interpreter or downloads one if absent.
macOS 14+ on Apple silicon and Linux with glibc 2.28+ are required by the selected PyTorch
wheels. Intel Macs, Windows ARM,
and Linux ARM are outside the initial supported platforms.

## Everyday commands

Examples below use `cpu`; NVIDIA users substitute `cuda`.

```sh
# Reproduce the committed environment, including development tools and notebooks
uv sync --locked --extra cpu --group notebooks

# Run a script with the project environment
uv run --locked --extra cpu python path/to/script.py

# Start notebooks; select the project's Python kernel
uv run --locked --extra cpu --group notebooks jupyter lab

# Add a shared dependency (updates pyproject.toml and uv.lock)
uv add --no-sync PACKAGE_NAME

# Add a development dependency
uv add --dev --no-sync PACKAGE_NAME

# Install the updated environment with your selected profile
uv sync --locked --extra cpu --group notebooks

# Quality checks
uv run --locked --extra cpu ruff check .
uv run --locked --extra cpu ruff format --check .
uv run --locked --extra cpu pytest

# Device checks; explicit unavailable devices fail rather than silently falling back
uv run --locked --extra cpu biomed-check --device cpu
uv run --locked --extra cpu biomed-check --device mps

# Use the project-local Hugging Face CLI
uv run --locked --extra cpu hf --help
```

`PACKAGE_NAME` and `path/to/script.py` above are placeholders. Commit `pyproject.toml` and `uv.lock`
together after changing dependencies. Use `uv add`, not ad hoc `pip install`, to keep them consistent.
The lockfile records exact versions; `--locked` prevents accidental dependency re-resolution.
You do not need to activate `.venv` when using `uv run`.

The existing global `hf` installation is independent of this repository. Use **`uv run … hf`**
here so it runs under the project's Python 3.12 environment. Public dataset metadata can be
queried anonymously. For gated/private resources, authenticate with `uv run --extra cpu hf auth login`;
never put tokens in notebooks, source files, or Git.

## MRI models and examples

The current MRI scripts are a small CrossPan example; they do not determine the project's
final dataset, modality, or models. To inspect the data, download only the scans and masks:

```sh
uv run --locked --extra cpu python scripts/download_mri_examples.py
```

To run the current UniverSeg example as well:

```sh
uv sync --locked --extra cpu --group models --group notebooks
uv run --locked --extra cpu python scripts/download_mri_examples.py --with-universeg-weights
uv run --locked --extra cpu --group models python scripts/run_mri_examples.py --model universeg
uv run --locked --extra cpu --group notebooks jupyter lab
```

NVIDIA users substitute `cuda` for `cpu`. The runner selects CUDA, MPS, or CPU automatically;
use `--device cpu`, `--device mps`, or `--device cuda` to choose explicitly.

The default download is ten CrossPan MRI/mask pairs (**38.4 MiB**) from a pinned snapshot.
`--with-universeg-weights` explicitly adds the official checkpoint (**4.5 MiB**, SHA256-verified).
Existing files are reused.
Public downloads need no Hugging Face login. CrossPan is
[CC BY-NC 4.0](https://huggingface.co/datasets/linkai-peng/CrossPan); see the
[UniverSeg model license](https://github.com/JJGO/UniverSeg/blob/833a0c34c65e38d675e21bd48ddec6797cc03259/LICENSE-model).
The download step performs no inference.

The [CrossPan inventory](data/crosspan_examples.csv) records scan facts: IDs, sequence,
source folder, patient group, and identity evidence. It assigns no model or experiment roles.
The current query cases and UniverSeg support cases are small lists at the top of
`scripts/run_mri_examples.py`; patient separation is checked when that example is run.
Other models can use the same scans differently.
The release provides sequence-specific split folders, but no complete patient crosswalk.
EMC groups use filename subject codes; CAD groups use original IDs from the upstream
[PanSegData T2 mapping](https://osf.io/download/9hrvs/). They are conservative exclusion groups,
not a verified paired cohort. Both query scans are previously inspected development cases.

The model step processes the two development queries sequentially, using **64 support
slices from four separate patients per sequence**. It calls the existing inference script
and saves predictions exactly where the notebook expects them. On the M1 pilot this took
about 70 seconds per query; CPU runs can take longer. Complete saved outputs are reused;
incomplete outputs produce a clear error so they can be moved aside before retrying.
Use `--dry-run` to inspect the commands without running models.

To also generate the optional TotalSegmentator comparator:

```sh
uv run --locked --extra cpu --group models python scripts/run_mri_examples.py --model totalsegmentator
```

This uses the full-resolution `total_mr` task and may download **442 MiB** of MRI weights
on first use, with additional space for extracted weights. Weights stay under `models/`.
Model libraries are pinned in the existing dependency group; these scripts add no dependencies.

Open [data-visualization-and-universeg.ipynb](notebooks/data-visualization-and-universeg.ipynb).
With data alone it shows images and annotations and reports missing predictions explicitly.
With generated predictions it also calculates whole-volume Dice and displays model overlays.
**The notebook needs no model weights to read saved predictions**, and performs no downloads
or inference itself. Data, weights, predictions, and executed outputs stay local and ignored by Git.

For a different query or support pool, `scripts/run_universeg.py --help` lists the explicit
image, mask, checkpoint, and output arguments. MRI preprocessing clips nonzero-voxel
percentiles at 0.5/99.5, normalizes each volume, and resizes images/masks to 128³. The default
is exactly 64 distinct processed support slices balanced across patients. Query labels are
never model inputs; probabilities return to the native grid before thresholding. Standardized
slices are interpolated views, not additional acquired scans.

## Working with notebooks and shared code

Use [environment-check.ipynb](notebooks/environment-check.ipynb) to verify the local setup.
JupyterLab launched through `uv run` can import `ml_biomed_project` without changing `sys.path`
or registering a global kernel. Open
[data-visualization-and-universeg.ipynb](notebooks/data-visualization-and-universeg.ipynb)
to inspect real MRI data, understand support sets, and evaluate the current 64-slice UniverSeg
run alongside TotalSegmentator, with per-case overlap and empty slice predictions made explicit.
Put reusable logic in `src/ml_biomed_project/`; use notebooks for exploration and figures.
Use **Restart Kernel and Clear Outputs of All Cells** before committing notebooks. CI checks
that committed notebooks have no outputs or execution counts.

## Project layout and collaboration

| Location | Purpose |
| --- | --- |
| `src/ml_biomed_project/` | Shared Python code and the `biomed-check` command |
| `notebooks/` | Exploration, visual inspection, and figures; committed without outputs |
| `tests/` | Small tests for shared behavior |
| `docs/` | Literature references |
| `data/`, `models/`, `results/` | Local datasets, checkpoints, and generated artifacts; ignored by Git |
| `pyproject.toml`, `uv.lock` | Dependency definitions and exact cross-platform resolutions |

Use short branches and pull requests for changes. Teammates need access to the private repository.
CI runs Ruff, tests, a real CPU check, and notebook-output checks on Linux, Windows, and macOS.
Hosted CI does not verify CUDA or MPS execution; run `biomed-check` on the intended accelerator.

Keep datasets, weights, and generated results out of Git. Preserve original data and keep
support and evaluation patients separate.
