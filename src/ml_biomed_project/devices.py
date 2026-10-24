"""Portable PyTorch device selection; no models or downloads are involved."""

import torch


def select_device(requested: str = "auto") -> torch.device:
    """Prefer CUDA, MPS, then CPU; explicit unavailable devices raise an error."""
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"Unknown device {requested!r}; choose auto, cpu, mps, or cuda.")

    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    if requested == "auto":
        return torch.device("cuda" if cuda_available else "mps" if mps_available else "cpu")
    if requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA is unavailable. On supported Linux/Windows NVIDIA machines, install the "
            "cuda profile and check the NVIDIA driver. Otherwise use --device cpu."
        )
    if requested == "mps" and not mps_available:
        raise RuntimeError(
            "MPS is unavailable. It requires an Apple silicon Mac and an MPS-enabled "
            "PyTorch build. If running in a restricted tool session, retry in Terminal."
        )
    return torch.device(requested)
