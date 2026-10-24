# ML in Biomedicine project

Medical image segmentation experiments for a three-person Aalto course project.
The final dataset, anatomy, and model comparison await supervisor feedback.
Pancreas segmentation across MRI sequences is a **proposed direction**, not a settled scope.
See [research notes](docs/research-notes.md) for the discussion and references.

This repository provides the environment, device checks, and a synthetic inspection example.
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

## Working with notebooks and shared code

Start with [00-environment.ipynb](notebooks/00-environment.ipynb). JupyterLab launched through
`uv run` can import `ml_biomed_project` without changing `sys.path` or registering a global kernel.
Then open [Inspect image and mask](notebooks/01-inspect-image-and-mask.ipynb) to check geometry
and view a synthetic annotation overlay. Its defaults download nothing. It uses the shared
`dice_score` function, which scores whole binary arrays and returns NaN for two empty masks.
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
