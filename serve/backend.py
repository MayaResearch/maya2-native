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
    """Detect PyTorch's active accelerator without assuming API == capability.

    An explicit torch_module is accepted to make detection unit-testable.
    """
    if torch_module is None:
        import torch as torch_module

    hip = getattr(getattr(torch_module, "version", None), "hip", None)
    cuda = getattr(getattr(torch_module, "version", None), "cuda", None)
    available = bool(torch_module.cuda.is_available())
    if hip:
        name = "rocm"
    elif cuda:
        name = "cuda"
    else:
        name = "cpu"

    device_name = None
    device_index = None
    if available:
        device_index = int(torch_module.cuda.current_device())
        device_name = str(torch_module.cuda.get_device_name(device_index))

    triton_importable = False
    if probe_triton:
        try:
            import triton  # noqa: F401
            triton_importable = True
        except Exception:
            pass

    return BackendInfo(
        name=name,
        available=available,
        torch_version=str(getattr(torch_module, "__version__", "unknown")),
        hip_version=str(hip) if hip else None,
        cuda_version=str(cuda) if cuda else None,
        device_name=device_name,
        device_index=device_index,
        triton_importable=triton_importable,
        # This only reports that the compatibility namespace exposes the symbol.
        # It is not proof that graph capture/replay works on the selected backend.
        cuda_graph_api_present=hasattr(torch_module.cuda, "CUDAGraph"),
    )
