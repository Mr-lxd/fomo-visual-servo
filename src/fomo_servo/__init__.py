"""Package boundary for the FOMO visual-servo project."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ConfigurationError",
    "ProjectConfig",
    "TrainingConfig",
    "load_config",
    "__version__",
]

__version__ = "0.0.0"


_CONFIG_EXPORTS = frozenset(
    {
        "ConfigurationError",
        "ProjectConfig",
        "TrainingConfig",
        "load_config",
    }
)


def __getattr__(name: str) -> Any:
    """Load YAML-backed configuration APIs only when a caller requests them."""

    if name not in _CONFIG_EXPORTS:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
    value = getattr(import_module(".config", __name__), name)
    globals()[name] = value
    return value
