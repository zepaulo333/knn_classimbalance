"""Central configuration loader.

All other modules import ``load_config()`` instead of hard-coding any
path, seed, or hyperparameter.  The YAML file is located at
``<project_root>/config/settings.yaml``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

# Project root is three levels up from this file:
#   src/utils/config.py  →  src/utils/  →  src/  →  project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SETTINGS_PATH = _PROJECT_ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def load_config(path: Path | None = None) -> dict:
    """Load and return the settings dictionary (cached after first call).

    Parameters
    ----------
    path : Path | None
        Override the default settings path (useful in tests).
    """
    target = Path(path) if path is not None else _SETTINGS_PATH
    with target.open("r") as fh:
        return yaml.safe_load(fh)


def get_project_root() -> Path:
    return _PROJECT_ROOT
