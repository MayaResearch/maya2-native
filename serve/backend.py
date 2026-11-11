"""Backend discovery and conservative capability reporting for Maya3.

This module intentionally reports API presence separately from validated support.
ROCm exposes the torch.cuda namespace; do not infer NVIDIA CUDA from that namespace.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BackendInfo:
    name: str
    available: bool
    torch_version: str | None
    hip_version: str | None
    cuda_version: str | None
    device_name: str | None
    device_index: int | None
    triton_importable: bool
    cuda_graph_api_present: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_backend(torch_module=None, *, probe_triton: bool = True) -> BackendInfo:
    """Detect build/backend identity separately from runtime device availability.

    An explicit torch_module is accepted to make detection unit-testable.
    A CUDA-built PyTorch without an accessible GPU remains backend="cuda",
    available=False. ROCm is identified by the HIP build marker, not by the
    torch.cuda namespace (which ROCm also exposes).
    """
    if torch_module is None:
        import torch as torch_module

    hip = getattr(getattr(torch_module, "version", None), "hip", None)
    cuda = getattr(getattr(torch_module, "version", None), "cuda", None)
    if hip:
        name = "rocm"
    elif cuda:
        name = "cuda"
    else:
        name = "cpu"

    available = bool(torch_module.cuda.is_available())
    device_name = None
    device_index = None
    if available:
        # Diagnostics must not make backend identification fail if device
        # metadata cannot be queried (e.g. a driver/runtime initialization issue).
        try:
            device_index = int(torch_module.cuda.current_device())
            device_name = str(torch_module.cuda.get_device_name(device_index))
        except Exception:
            device_index = None
            device_name = None

    triton_importable = False
    if probe_triton:
        try:
            import triton  # noqa: F401
            triton_importable = True
        except Exception:
            pass

    cuda_namespace = getattr(torch_module, "cuda", None)
    return BackendInfo(
        name=name,
        available=available,
        torch_version=str(getattr(torch_module, "__version__", "unknown")),
        hip_version=str(hip) if hip else None,
        cuda_version=str(cuda) if cuda else None,
        device_name=device_name,
        device_index=device_index,
        triton_importable=triton_importable,
        # Symbol presence is diagnostic only, not proof of support/correctness.
        cuda_graph_api_present=hasattr(cuda_namespace, "CUDAGraph"),
    )
