"""Bundle launcher: run deployment CLIs without manual PYTHONPATH setup.

The launcher must sit at the bundle root next to ``src/`` (the repository root
has the same layout, so the same file works in both places). It resolves the
bundle root from its own file location, prepends ``<root>/src`` to
``sys.path``, and dispatches to one allowlisted ``scripts.*`` CLI module. Only
the standard library is used, so it works in the minimal Pi runtime without
torch or any training dependency.

Usage::

    python run.py <entry> [arguments...]

Entries: predict_image, predict_video, capture_dataset.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


ENTRY_POINTS = {
    "predict_image": "scripts.predict_image",
    "predict_video": "scripts.predict_video",
    "capture_dataset": "scripts.capture_dataset",
}

USAGE = (
    "usage: python run.py {" + "|".join(sorted(ENTRY_POINTS)) + "} [arguments...]"
)


def bundle_root() -> Path:
    """Return the bundle root: the directory that contains this launcher."""

    return Path(__file__).resolve().parent


def bootstrap_sys_path(root: Path) -> None:
    """Prepend ``<root>/src`` so ``scripts.*`` and ``fomo_servo`` import."""

    source_directory = root / "src"
    if not source_directory.is_dir():
        print(
            "Error: bundle layout is invalid; expected src/ next to {}".format(
                root / "run.py"
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)
    sys.path.insert(0, str(source_directory))


def main(argv: list[str]) -> int:
    arguments = argv[1:]
    if not arguments:
        print(USAGE)
        return 2
    if arguments[0] in ("-h", "--help"):
        print(__doc__.strip())
        print()
        print(USAGE)
        return 0
    name = arguments[0]
    if name not in ENTRY_POINTS:
        print(
            "Error: unknown entry '{}'; available entries: {}".format(
                name, ", ".join(sorted(ENTRY_POINTS))
            ),
            file=sys.stderr,
        )
        return 2
    bootstrap_sys_path(bundle_root())
    module = importlib.import_module(ENTRY_POINTS[name])
    return module.main(arguments[1:])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
