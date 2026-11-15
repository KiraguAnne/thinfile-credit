"""Config loading and shared paths."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path or os.environ.get("THINFILE_CONFIG", DEFAULT_CONFIG))
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    for key, val in cfg["paths"].items():
        if key.endswith("_dir"):
            p = Path(val)
            cfg["paths"][key] = p if p.is_absolute() else ROOT / p
    return cfg


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s", datefmt="%H:%M:%S"
    )
    return logging.getLogger(name)
