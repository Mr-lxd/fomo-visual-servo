"""Preflight protection against inference output/input path aliasing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


class OutputPathError(ValueError):
    """Raised before an output could overwrite a protected or sibling artifact."""


def validate_output_paths(
    *,
    protected_inputs: Mapping[str, Path],
    outputs: Mapping[str, Path],
) -> None:
    """Require every output to be distinct from all inputs and other outputs."""

    normalized_inputs = {
        name: _normalized_path(Path(path)) for name, path in protected_inputs.items()
    }
    normalized_outputs: dict[str, tuple[Path, str]] = {}
    for output_name, output_path in outputs.items():
        output = Path(output_path)
        _validate_win32_output_components(output, output_name)
        candidate = _normalized_path(output)
        for input_name, protected in normalized_inputs.items():
            if _paths_alias(candidate, protected):
                raise OutputPathError(
                    "output '{}' aliases protected input '{}'".format(
                        output_name, input_name
                    )
                )
        for sibling_name, sibling in normalized_outputs.items():
            if _paths_alias(candidate, sibling):
                raise OutputPathError(
                    "output '{}' aliases output '{}'".format(
                        output_name, sibling_name
                    )
                )
        normalized_outputs[output_name] = candidate


def _validate_win32_output_components(path: Path, output_name: str) -> None:
    if os.name != "nt":
        return
    anchor = path.anchor
    for component in path.parts:
        if component == anchor or component in {".", ".."}:
            continue
        if component.endswith((".", " ")):
            raise OutputPathError(
                "output '{}' uses unsafe Win32 output path component {!r} ending "
                "in a dot or space".format(output_name, component)
            )


def _normalized_path(path: Path) -> tuple[Path, str]:
    try:
        resolved = path.resolve(strict=False)
    except OSError as error:
        raise OutputPathError("unable to resolve path '{}': {}".format(path, error)) from error
    return resolved, os.path.normcase(str(resolved))


def _paths_alias(
    left: tuple[Path, str], right: tuple[Path, str]
) -> bool:
    left_path, left_text = left
    right_path, right_text = right
    if left_text == right_text:
        return True
    if left_path.exists() and right_path.exists():
        try:
            return os.path.samefile(left_path, right_path)
        except OSError as error:
            raise OutputPathError(
                "unable to compare paths '{}' and '{}': {}".format(
                    left_path, right_path, error
                )
            ) from error
    return False


__all__ = ["OutputPathError", "validate_output_paths"]
