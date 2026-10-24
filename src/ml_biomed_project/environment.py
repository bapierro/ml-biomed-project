"""Report the environment and execute a tiny calculation on the selected device."""

import argparse
import json
import platform
import sys
from importlib.metadata import version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()

    try:
        import torch

        from ml_biomed_project.devices import select_device
    except ImportError:
        parser.exit(1, "Install a PyTorch profile first: uv sync --extra cpu (or --extra cuda).\n")

    try:
        device = select_device(args.device)
        with torch.inference_mode():
            x = torch.ones((4, 4), device=device)
            result = (x @ x).cpu()
        if not torch.equal(result, torch.full((4, 4), 4.0)):
            raise RuntimeError("The device calculation returned an unexpected result.")
    except (RuntimeError, ValueError) as exc:
        print(f"Environment check failed: {exc}", file=sys.stderr)
        return 1

    packages = ("numpy", "scipy", "pandas", "matplotlib", "nibabel", "huggingface-hub", "torch")
    report = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "packages": {name: version(name) for name in packages},
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "selected_device": str(device),
        "calculation": "passed",
    }
    print(json.dumps(report, indent=2))
    return 0
