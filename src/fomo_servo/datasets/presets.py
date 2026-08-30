"""Canonical augmentation preset definitions and override resolution."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


PRESET_DEFINITIONS: dict[str, dict[str, Any]] = {
    "none": {
        "color_jitter": {"enabled": False, "probability": 0.0},
        "horizontal_flip": {"enabled": False, "probability": 0.0},
        "gaussian_blur": {"enabled": False, "probability": 0.0},
        "gaussian_noise": {"enabled": False, "probability": 0.0},
        "affine": {"enabled": False, "probability": 0.0},
    },
    "photometric": {
        "color_jitter": {
            "enabled": True,
            "probability": 0.8,
            "brightness": 0.20,
            "contrast": 0.20,
            "saturation": 0.20,
            "hue": 0.02,
        },
        "horizontal_flip": {"enabled": False, "probability": 0.0},
        "gaussian_blur": {
            "enabled": True,
            "probability": 0.15,
            "kernel_sizes": [3, 5],
            "sigma_min": 0.1,
            "sigma_max": 1.0,
        },
        "gaussian_noise": {
            "enabled": True,
            "probability": 0.15,
            "std_min": 2.0,
            "std_max": 8.0,
        },
        "affine": {"enabled": False, "probability": 0.0},
    },
    "underwater_conservative": {
        "color_jitter": {
            "enabled": True,
            "probability": 0.8,
            "brightness": 0.20,
            "contrast": 0.20,
            "saturation": 0.20,
            "hue": 0.02,
        },
        "horizontal_flip": {"enabled": True, "probability": 0.5},
        "gaussian_blur": {
            "enabled": True,
            "probability": 0.15,
            "kernel_sizes": [3, 5],
            "sigma_min": 0.1,
            "sigma_max": 1.0,
        },
        "gaussian_noise": {
            "enabled": True,
            "probability": 0.15,
            "std_min": 2.0,
            "std_max": 8.0,
        },
        "affine": {
            "enabled": True,
            "probability": 0.30,
            "scale_min": 0.90,
            "scale_max": 1.10,
            "translate_fraction": 0.05,
            "rotation_degrees": 5.0,
            "min_visibility": 0.25,
            "border_value": 114,
        },
    },
    "custom": {
        "color_jitter": {},
        "horizontal_flip": {},
        "gaussian_blur": {},
        "gaussian_noise": {},
        "affine": {},
    },
}

KNOWN_AUGMENTATION_FIELDS = frozenset(
    {
        "enabled",
        "preset",
        "overrides",
        "color_jitter",
        "horizontal_flip",
        "gaussian_blur",
        "gaussian_noise",
        "affine",
    }
)


def resolve_preset_mapping(
    preset: str, overrides: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return a deep-copied, validated primitive mapping for one preset."""

    if preset not in PRESET_DEFINITIONS:
        raise ValueError(
            "unknown augmentation preset '{}'; expected one of {}".format(
                preset, sorted(PRESET_DEFINITIONS)
            )
        )
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise ValueError("augmentation.overrides must be a mapping")
    resolved = deepcopy(PRESET_DEFINITIONS[preset])
    for raw_path, value in overrides.items():
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("augmentation override paths must be non-empty strings")
        parts = raw_path.split(".")
        if len(parts) != 2 or parts[0] not in resolved:
            raise ValueError("unknown augmentation override field '{}'".format(raw_path))
        operation, field = parts
        if field not in _KNOWN_OPERATION_FIELDS[operation]:
            raise ValueError("unknown augmentation override field '{}'".format(raw_path))
        resolved.setdefault(operation, {})[field] = deepcopy(value)
    return resolved


_KNOWN_OPERATION_FIELDS = {
    "color_jitter": {"enabled", "probability", "brightness", "contrast", "saturation", "hue"},
    "horizontal_flip": {"enabled", "probability"},
    "gaussian_blur": {"enabled", "probability", "kernel_sizes", "sigma_min", "sigma_max"},
    "gaussian_noise": {"enabled", "probability", "std_min", "std_max"},
    "affine": {
        "enabled",
        "probability",
        "scale_min",
        "scale_max",
        "translate_fraction",
        "rotation_degrees",
        "min_visibility",
        "border_value",
    },
}


def known_operation_fields(name: str) -> frozenset[str]:
    """Return the allowed primitive fields for an operation."""

    try:
        return frozenset(_KNOWN_OPERATION_FIELDS[name])
    except KeyError as error:
        raise ValueError("unknown augmentation operation '{}'".format(name)) from error


__all__ = [
    "KNOWN_AUGMENTATION_FIELDS",
    "PRESET_DEFINITIONS",
    "known_operation_fields",
    "resolve_preset_mapping",
]
