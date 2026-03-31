import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory for a file path."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def save_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    """Persist a dataframe to CSV."""
    file_path = ensure_parent(path)
    df.to_csv(file_path, index=index)


def save_json(payload: Any, path: str | Path) -> None:
    """Persist a JSON-serializable object."""
    file_path = ensure_parent(path)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

