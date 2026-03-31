import sys
from typing import Any

import torch


EXPECTED_GPU_SUBSTRING = "A100"


def runtime_summary() -> dict[str, Any]:
    """Collect the active PyTorch runtime details."""
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None

    return {
        "python": sys.version.split("\n")[0],
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
    }


def require_a100() -> torch.device:
    """Return the CUDA device and fail fast if Colab is not on an A100."""
    summary = runtime_summary()

    if not summary["cuda_available"]:
        raise RuntimeError(
            "CUDA is not available. In Google Colab, switch Runtime > Change runtime type > Hardware accelerator > GPU, then reconnect."
        )

    gpu_name = summary["gpu_name"] or ""
    if EXPECTED_GPU_SUBSTRING not in gpu_name:
        raise RuntimeError(
            f"Expected a GPU containing '{EXPECTED_GPU_SUBSTRING}', but found '{gpu_name}'. Reconnect this notebook to an A100 runtime before training."
        )

    return torch.device("cuda")


def print_runtime_summary() -> torch.device:
    """Print a readable runtime summary and return the selected device."""
    summary = runtime_summary()

    print(f"python: {summary['python']}")
    print(f"torch: {summary['torch']}")
    print(f"cuda_available: {summary['cuda_available']}")
    print(f"cuda_version: {summary['cuda_version']}")
    print(f"gpu_name: {summary['gpu_name']}")
    print(f"device_count: {summary['device_count']}")

    device = require_a100()
    print(f"selected_device: {device}")
    return device
