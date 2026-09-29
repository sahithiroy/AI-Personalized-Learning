"""Configuration loading (config.yaml + environment variables)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    path = Path(os.getenv("PLRS_CONFIG", PROJECT_ROOT / "config.yaml"))
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Environment overrides for the most commonly changed values
    if os.getenv("PLRS_PRIMARY_LLM"):
        cfg["llm"]["primary"] = os.environ["PLRS_PRIMARY_LLM"]
    return cfg


def resolve_path(relative: str) -> Path:
    p = Path(relative)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def store_dir() -> Path:
    return resolve_path(get_config()["paths"]["store_dir"])


def models_dir() -> Path:
    return resolve_path(get_config()["paths"]["models_dir"])
