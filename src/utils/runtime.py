import shutil
import subprocess
from typing import Any

import numpy as np
import torch


def get_runtime_context() -> dict[str, Any]:
    """Collect lightweight runtime context information."""
    context: dict[str, Any] = {
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        "gpu_name": None,
        "driver_version": None,
    }

    if context["nvidia_smi_available"]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            first_line = result.stdout.strip().splitlines()[0]
            gpu_name, driver_version = [item.strip() for item in first_line.split(",", 1)]
            context["gpu_name"] = gpu_name
            context["driver_version"] = driver_version
        except Exception:
            pass

    return context


def select_torch_device() -> torch.device:
    """Choose the best available torch device in a stable order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_global_seed(seed: int) -> None:
    """Seed Python numerical backends for reproducible runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_peak_gpu_memory_mb(device: torch.device) -> float | None:
    """Return peak allocated CUDA memory in MB when available."""
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))
