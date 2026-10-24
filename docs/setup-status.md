# Initial setup verification

Verified on **24 October 2026** on an Apple M1 MacBook Air with 16 GB unified memory,
macOS 26.2, and arm64 Python. No datasets, label volumes, pretrained weights, or CUDA
package wheels were downloaded. Only software packages and public repository metadata were retrieved.

## Installed tools and environment

| Component | Version |
| --- | --- |
| uv | 0.13.5; upgraded from Homebrew 0.9.7 |
| Python | 3.12.13; reused existing Homebrew interpreter |
| PyTorch | 2.11.0; CPU/MPS Mac build |
| NumPy | 2.5.4 |
| SciPy | 1.18.1 |
| pandas | 3.1.0 |
| Matplotlib | 3.11.2 |
| NiBabel | 5.4.2 |
| Hugging Face Hub / project-local hf | 2.3.0 |
| JupyterLab | 4.6.4 |
| ipykernel | 7.4.0 |
| pytest | 9.1.1 |
| Ruff | 0.17.2 |

The environment contains 121 installed distributions, including this project and transitive
dependencies. `uv.lock` includes additional packages for the other supported platforms/profiles;
their presence in the lockfile does not mean they were installed on the Mac.

The global Homebrew `hf` 2.2.0 installation was left unchanged. It fails during macOS detection
under its Python 3.15 runtime in the tested environment. The project-local Python 3.12 CLI works.
GitHub CLI authentication was verified using the existing saved account.

## Checks performed

- Locked environment installation and `uv pip check`: passed.
- Ruff lint and formatting checks: passed.
- Nine device-selection/CPU tests: passed.
- Real CPU and MPS matrix calculations through `biomed-check`: passed.
- Notebook executed in a real Python kernel from `notebooks/`: passed; shared-package imports
  worked, the kernel used `.venv/bin/python`, and automatic selection chose MPS.
- JupyterLab started on localhost; its authenticated `/lab` page returned HTTP 200; the
  temporary server was stopped afterward.
- Notebook outputs and execution counts remain clear in the saved file.
- Explicitly requesting CUDA on the Mac returned an unavailable-device error and exit status 1.
- Locked installation dry runs for Linux x86_64 and Windows x86_64 succeeded for both profiles:
  `torch==2.11.0+cpu` and `torch==2.11.0+cu128` respectively. No foreign-platform wheels were fetched.
- The project-local `hf datasets info` command and Python API both successfully accessed public
  CrossPan metadata anonymously. Observed dataset revision:
  `0da3821e8af2dfa4c22d56e282372614cb426227`. This is a metadata observation, not a dataset download
  or a decision to use this revision for experiments.
- Git ignore rules were checked for datasets, weights, generated results, environments,
  credentials, and notebook checkpoints.

GitHub Actions runs CPU checks on Linux, Windows, and Apple silicon macOS for pushes and pull
requests. See the repository's Actions tab for current remote run results.

## Storage

- The selected 120 wheel archives total **209,958,494 bytes (200.2 MiB)**; this excludes the tiny
  editable project build. This is the sum of package archive sizes, not a network traffic meter.
- The local `.venv` used approximately **855 MiB** after execution/import checks, including
  generated Python bytecode. Actual size can change as more packages are imported.
- Homebrew reports the upgraded uv installation as **33.9 MB**.
- The shared uv cache is outside this folder and can also use disk space. It is reusable across
  projects; its size is not included in `.venv` above. Files may share storage, so directory sizes
  should not be added as if they were all independent physical copies.
- Approximately **44 GiB** remained free after initial verification. Local `data/`, `models/`,
  and `results/` contain only placeholders.

## Verification limits and next milestone

CUDA computation has not been tested because this Mac has no NVIDIA GPU. A teammate should run
`uv run --locked --extra cuda biomed-check --device cuda` on a compatible Linux/Windows machine
before starting experiments. Installing a CUDA build does not establish that a particular GPU or
driver is compatible. No model-specific compatibility or inference performance is claimed yet.

The next research milestone is to confirm the topic with the supervisor, choose a small pilot,
and agree on exact data/checkpoint files and download sizes before retrieving them.
