"""Device selection is tested independently of available accelerator hardware."""

import pytest
import torch

from ml_biomed_project.devices import select_device


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_device_priority(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert select_device().type == expected


@pytest.mark.parametrize("requested", ["mps", "cuda"])
def test_explicit_unavailable_device_fails(monkeypatch, requested):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        select_device(requested)


def test_cpu_can_be_requested_with_accelerators_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert select_device("cpu").type == "cpu"


def test_unknown_device_fails():
    with pytest.raises(ValueError, match="Unknown device"):
        select_device("tpu")


def test_real_cpu_calculation():
    x = torch.ones((4, 4), device=select_device("cpu"))
    assert torch.equal(x @ x, torch.full((4, 4), 4.0))
