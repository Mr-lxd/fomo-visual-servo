"""Package boundary for the FOMO visual-servo project."""

from .config import ConfigurationError, ProjectConfig, TrainingConfig, load_config

__all__ = [
    "ConfigurationError",
    "ProjectConfig",
    "TrainingConfig",
    "load_config",
    "__version__",
]

__version__ = "0.0.0"
