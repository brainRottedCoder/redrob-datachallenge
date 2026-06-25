"""Shared configuration helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_WEIGHTS_PATH = Path("config/weights.yaml")


def load_weights(path: str | Path = DEFAULT_WEIGHTS_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
