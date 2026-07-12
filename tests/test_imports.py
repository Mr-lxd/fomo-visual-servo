from __future__ import annotations

import importlib

import pytest


PACKAGE_MODULES = (
    "fomo_servo",
    "fomo_servo.config",
    "fomo_servo.datasets",
    "fomo_servo.datasets.augmentation",
    "fomo_servo.models",
    "fomo_servo.training",
    "fomo_servo.losses",
    "fomo_servo.metrics",
    "fomo_servo.postprocess",
    "fomo_servo.inference",
    "fomo_servo.export",
)


@pytest.mark.parametrize("module_name", PACKAGE_MODULES)
def test_skeleton_package_modules_import(module_name: str) -> None:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        module = None

    assert module is not None, f"{module_name} must be importable"
