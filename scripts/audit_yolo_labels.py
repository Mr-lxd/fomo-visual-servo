"""Create immutable parity-clean audit artifacts from a YOLO dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from fomo_servo.config import ConfigurationError, load_config
from fomo_servo.evaluation.parity_clean import (
    ParityCleanError,
    audit_yolo_dataset,
    build_parity_clean_manifest,
    write_audit_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the read-only audit CLI parser."""

    parser = argparse.ArgumentParser(
        description="Audit YOLO labels and write a manifest-only parity-clean view."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--dataset-root", type=Path)
    parser.add_argument("--class-count", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(arguments: Optional[Sequence[str]] = None) -> int:
    """Run the immutable audit and print JSON paths/hashes to stdout."""

    args = build_parser().parse_args(arguments)
    try:
        if args.config is not None:
            config = load_config(args.config)
            dataset_root = config.dataset.root
            class_count = len(config.dataset.class_names)
            if args.class_count is not None and args.class_count != class_count:
                raise ParityCleanError(
                    "--class-count does not match the configured class table"
                )
        else:
            dataset_root = args.dataset_root
            class_count = args.class_count
            if class_count is None:
                raise ParityCleanError("--class-count is required with --dataset-root")
        audit = audit_yolo_dataset(dataset_root, class_count=class_count)
        manifest = build_parity_clean_manifest(audit)
        json_path, csv_path, manifest_path = write_audit_artifacts(
            args.output_dir, audit, manifest
        )
        print(
            json.dumps(
                {
                    "invalid_label_audit_json": str(json_path),
                    "invalid_label_audit_csv": str(csv_path),
                    "parity_clean_manifest": str(manifest_path),
                    "cleaning_view_hash": manifest["cleaning_view_hash"],
                    "cleaned_test_view_hash": manifest["cleaned_test_view_hash"],
                    "physical_invalid_row_count": audit.physical_invalid_row_count,
                },
                ensure_ascii=False,
            )
        )
    except (ConfigurationError, ParityCleanError, OSError, ValueError) as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
