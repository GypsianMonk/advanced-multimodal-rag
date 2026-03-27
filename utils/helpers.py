"""utils/helpers.py — Config loading and general utilities."""

from __future__ import annotations

import yaml
from pathlib import Path


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
