"""Strict, manifest-only YOLO label views for parity evaluation.

The source dataset remains read-only. A :class:`ParityCleanView` verifies the
hash of every accessed label file, removes only rows declared in its manifest,
then uses the ordinary YOLO parser to reject any remaining invalid row.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from fomo_servo.datasets.yolo import NormalizedYoloBox, YoloLabelError, parse_yolo_label_lines


class ParityCleanError(ValueError):
    """Raised when an audit or declared cleaned view cannot be trusted."""


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_REMOVABLE_CATEGORIES = frozenset({"width_lte_zero", "height_lte_zero"})
_INVALID_CATEGORIES = frozenset(
    {
        "invalid_field_count",
        "invalid_numeric",
        "nan_or_inf",
        "invalid_class_id",
        "width_lte_zero",
        "height_lte_zero",
        "out_of_bounds",
    }
)


@dataclass(frozen=True)
class LabelAudit:
    """Read-only audit details for all physical dataset splits."""

    dataset_root: Path
    class_count: int
    issues: tuple[dict[str, object], ...]
    summary_by_split: Mapping[str, Mapping[str, int]]
    split_counts: Mapping[str, Mapping[str, object]]
    physical_invalid_rows: tuple[dict[str, object], ...]
    undeclared_invalid_rows: tuple[dict[str, object], ...]

    @property
    def physical_invalid_row_count(self) -> int:
        """Return the unique number of invalid non-empty annotation rows."""

        return len(self.physical_invalid_rows)


def audit_yolo_dataset(dataset_root: Path, class_count: int) -> LabelAudit:
    """Audit every physical split without changing image or label bytes.

    The return records raw content only for problematic non-empty lines. Empty
    labels are reported as no-object samples, not as invalid annotations.
    """

    root = Path(dataset_root)
    if not root.is_dir():
        raise ParityCleanError("dataset root does not exist: {}".format(root))
    if isinstance(class_count, bool) or not isinstance(class_count, int) or class_count <= 0:
        raise ParityCleanError("class_count must be a positive integer")
    split_names = _find_split_names(root)
    issues: list[dict[str, object]] = []
    physical_rows: list[dict[str, object]] = []
    split_counts: dict[str, Mapping[str, object]] = {}
    summaries: dict[str, Mapping[str, int]] = {}

    for split in split_names:
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        images = tuple(sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES))
        labels = tuple(sorted(path for path in label_dir.rglob("*.txt") if path.is_file()))
        image_keys = {path.relative_to(image_dir).with_suffix("").as_posix(): path for path in images}
        label_keys = {path.relative_to(label_dir).with_suffix("").as_posix(): path for path in labels}
        counts: Counter[str] = Counter()
        rows_before = 0
        rows_after_zero_area = 0
        per_class_before: Counter[str] = Counter()
        per_class_after: Counter[str] = Counter()

        for key, image_path in image_keys.items():
            if key not in label_keys:
                entry = _issue(split, "missing_label", image_path, root)
                entry["expected_label_relative_path"] = (Path(split) / "labels" / (key + ".txt")).as_posix()
                issues.append(entry)
                counts["missing_label"] += 1

        for key, label_path in label_keys.items():
            if key not in image_keys:
                issues.append(_issue(split, "orphan_label", label_path, root))
                counts["orphan_label"] += 1
            lines = label_path.read_text(encoding="utf-8-sig", errors="surrogateescape").splitlines()
            if not any(line.strip() for line in lines):
                issues.append(_issue(split, "empty_label", label_path, root))
                counts["empty_label"] += 1
                continue
            seen: dict[tuple[object, ...], int] = {}
            for line_number, raw in enumerate(lines, start=1):
                if not raw.strip():
                    continue
                row = _audit_row(raw, class_count)
                for category in row["categories"]:
                    issue = _issue(split, str(category), label_path, root, line_number, raw)
                    issue.update(row["details"])
                    issues.append(issue)
                    counts[str(category)] += 1
                if row["parsed"]:
                    class_id, x_center, y_center, width, height = row["parsed"]
                    duplicate_key = (class_id, x_center, y_center, width, height)
                    if duplicate_key in seen:
                        issue = _issue(split, "duplicate_bbox", label_path, root, line_number, raw)
                        issue["duplicate_of_line"] = seen[duplicate_key]
                        issues.append(issue)
                        counts["duplicate_bbox"] += 1
                    else:
                        seen[duplicate_key] = line_number
                    rows_before += 1
                    if 0 <= class_id < class_count:
                        per_class_before[str(class_id)] += 1
                    if width <= 0.0 or height <= 0.0:
                        pass
                    else:
                        rows_after_zero_area += 1
                        if 0 <= class_id < class_count:
                            per_class_after[str(class_id)] += 1
                if any(category in _INVALID_CATEGORIES for category in row["categories"]):
                    physical = _issue(split, "physical_invalid_row", label_path, root, line_number, raw)
                    physical["categories"] = list(row["categories"])
                    physical.update(row["details"])
                    physical_rows.append(physical)
        summaries[split] = dict(sorted(counts.items()))
        split_counts[split] = {
            "image_count": len(images),
            "label_file_count": len(labels),
            "gt_rows_before": rows_before,
            "gt_rows_after_zero_area_removal": rows_after_zero_area,
            "per_class_before": dict(sorted(per_class_before.items())),
            "per_class_after_zero_area_removal": dict(sorted(per_class_after.items())),
        }

    undeclared = tuple(
        row
        for row in physical_rows
        if not set(row["categories"]).issubset(_REMOVABLE_CATEGORIES)
    )
    return LabelAudit(
        dataset_root=root,
        class_count=class_count,
        issues=tuple(issues),
        summary_by_split=summaries,
        split_counts=split_counts,
        physical_invalid_rows=tuple(physical_rows),
        undeclared_invalid_rows=undeclared,
    )


def build_parity_clean_manifest(audit: LabelAudit) -> dict[str, object]:
    """Build a deterministic view that removes only audited zero-area rows."""

    if audit.undeclared_invalid_rows:
        raise ParityCleanError("cannot build parity-clean manifest with non-removable invalid rows")
    removed_rows = []
    removals: dict[str, set[int]] = defaultdict(set)
    for row in audit.physical_invalid_rows:
        categories = set(row["categories"])
        if not categories.issubset(_REMOVABLE_CATEGORIES):
            raise ParityCleanError("manifest removal contains a non-removable invalid category")
        relative_path = str(row["relative_path"])
        line_number = int(row["line_number"])
        removals[relative_path].add(line_number)
        removed_rows.append(
            {
                "relative_path": relative_path,
                "line_number": line_number,
                "raw_content": row["raw_content"],
                "reason": "width_lte_zero_and_height_lte_zero" if categories == _REMOVABLE_CATEGORIES else next(iter(categories)),
                "source_label_sha256": row["file_sha256"],
            }
        )
    source_files: dict[str, dict[str, object]] = {}
    for path in _view_file_paths(audit.dataset_root, audit.split_counts):
        relative_path = path.relative_to(audit.dataset_root).as_posix()
        source = path.read_bytes()
        virtual = _virtual_bytes(path, removals.get(relative_path, set()))
        source_files[relative_path] = {
            "sha256": _sha256(source),
            "size": len(source),
            "cleaned_sha256": _sha256(virtual),
            "cleaned_size": len(virtual),
        }
    all_hash = _content_hash(source_files, cleaned=True)
    test_files = {key: value for key, value in source_files.items() if key == "data.yaml" or key.startswith("test/")}
    test_hash = _content_hash(test_files, cleaned=True)
    for row in removed_rows:
        row["cleaned_label_sha256"] = source_files[str(row["relative_path"])]["cleaned_sha256"]
    return {
        "protocol": "parity-clean-v1",
        "source_dataset_read_only": True,
        "deletion_rule": "width <= 0 OR height <= 0",
        "class_count": audit.class_count,
        "removed_rows": removed_rows,
        "source_files": source_files,
        "cleaning_view_hash": all_hash,
        "cleaned_test_view_hash": test_hash,
        "audit_summary_by_split": audit.summary_by_split,
        "audit_split_counts": audit.split_counts,
    }


class ParityCleanView:
    """Read verified virtual label files specified by a parity-clean manifest."""

    def __init__(self, dataset_root: Path, manifest: Mapping[str, object], class_count: int) -> None:
        self.root = Path(dataset_root)
        self.manifest = dict(manifest)
        self.class_count = class_count
        if self.manifest.get("protocol") != "parity-clean-v1":
            raise ParityCleanError("unsupported cleaning manifest protocol")
        if self.manifest.get("source_dataset_read_only") is not True:
            raise ParityCleanError("cleaning manifest must declare a read-only source dataset")
        if self.manifest.get("class_count") != class_count:
            raise ParityCleanError("cleaning manifest class_count does not match dataset")
        raw_files = self.manifest.get("source_files")
        raw_rows = self.manifest.get("removed_rows")
        if not isinstance(raw_files, Mapping) or not isinstance(raw_rows, Sequence):
            raise ParityCleanError("cleaning manifest has invalid source_files or removed_rows")
        self.source_files = raw_files
        self.removals: dict[str, set[int]] = defaultdict(set)
        for row in raw_rows:
            if not isinstance(row, Mapping):
                raise ParityCleanError("cleaning manifest removed_rows must contain mappings")
            relative_path = row.get("relative_path")
            line_number = row.get("line_number")
            if not isinstance(relative_path, str) or not isinstance(line_number, int) or line_number <= 0:
                raise ParityCleanError("cleaning manifest removal has invalid path or line number")
            self.removals[relative_path].add(line_number)

    def read_label_lines(self, relative_path: str) -> tuple[str, ...]:
        """Return virtual rows after source verification and strict validation."""

        path = self._resolve_relative_path(relative_path)
        source = path.read_bytes()
        expected = self.source_files.get(relative_path)
        if not isinstance(expected, Mapping):
            raise ParityCleanError("label path is absent from the cleaning manifest: {}".format(relative_path))
        if _sha256(source) != expected.get("sha256"):
            raise ParityCleanError("source label SHA-256 does not match cleaning manifest: {}".format(relative_path))
        lines = path.read_text(encoding="utf-8-sig", errors="surrogateescape").splitlines()
        retained = tuple(line for number, line in enumerate(lines, start=1) if number not in self.removals.get(relative_path, set()))
        try:
            parse_yolo_label_lines(path, retained, self.class_count)
        except YoloLabelError as error:
            raise ParityCleanError("invalid YOLO row outside the manifest: {}".format(error)) from error
        # Preserve source bytes for untouched labels. For a declared removal,
        # `_virtual_bytes` uses the manifest's deterministic retained-row form.
        # This is intentionally the same byte rule used when the approved view
        # hash was built; parsing itself remains newline-format agnostic.
        virtual = _virtual_bytes(path, self.removals.get(relative_path, set()))
        if _sha256(virtual) != expected.get("cleaned_sha256"):
            raise ParityCleanError("cleaned label SHA-256 does not match cleaning manifest: {}".format(relative_path))
        return retained

    def parse_label_file(self, label_path: Path, class_count: int) -> tuple[NormalizedYoloBox, ...]:
        """Dataset-compatible label loader for the explicit cleaned view."""

        if class_count != self.class_count:
            raise ParityCleanError("dataset class count does not match cleaning manifest")
        path = Path(label_path)
        if not path.exists():
            return ()
        relative_path = path.relative_to(self.root).as_posix()
        return parse_yolo_label_lines(path, self.read_label_lines(relative_path), class_count)

    def _resolve_relative_path(self, relative_path: str) -> Path:
        candidate = (self.root / Path(relative_path)).resolve()
        try:
            candidate.relative_to(self.root.resolve())
        except ValueError as error:
            raise ParityCleanError("cleaning manifest path escapes dataset root") from error
        if not candidate.is_file():
            raise ParityCleanError("manifest label file does not exist: {}".format(relative_path))
        return candidate


def verify_parity_clean_view(
    dataset_root: Path, manifest: Mapping[str, object], class_count: int
) -> dict[str, str]:
    """Verify every parity-view input before a fixed test evaluation starts.

    The source dataset is never edited. This function hashes each declared
    image, label, and root ``data.yaml``; validates each virtual label file; and
    recomputes the complete and test-only cleaning-view hashes. Any source drift,
    added/removed consumed input, or invalid label outside the manifest raises
    :class:`ParityCleanError` before inference begins.
    """

    view = ParityCleanView(dataset_root, manifest, class_count)
    raw_split_counts = manifest.get("audit_split_counts")
    if not isinstance(raw_split_counts, Mapping):
        raise ParityCleanError("cleaning manifest audit_split_counts must be a mapping")
    expected_paths = {
        path.relative_to(view.root).as_posix()
        for path in _view_file_paths(view.root, raw_split_counts)
    }
    if set(view.source_files) != expected_paths:
        raise ParityCleanError("consumed dataset files do not exactly match cleaning manifest")

    verified_files: dict[str, dict[str, object]] = {}
    for relative_path in sorted(expected_paths):
        record = view.source_files.get(relative_path)
        if not isinstance(record, Mapping):
            raise ParityCleanError("cleaning manifest source file record is invalid: {}".format(relative_path))
        path = view._resolve_relative_path(relative_path)
        source = path.read_bytes()
        if _sha256(source) != record.get("sha256"):
            raise ParityCleanError("source SHA-256 does not match cleaning manifest: {}".format(relative_path))
        virtual = _virtual_bytes(path, view.removals.get(relative_path, set()))
        if _sha256(virtual) != record.get("cleaned_sha256"):
            raise ParityCleanError("cleaned SHA-256 does not match cleaning manifest: {}".format(relative_path))
        if len(source) != record.get("size") or len(virtual) != record.get("cleaned_size"):
            raise ParityCleanError("source or cleaned size does not match cleaning manifest: {}".format(relative_path))
        if path.suffix.lower() == ".txt":
            view.read_label_lines(relative_path)
        verified_files[relative_path] = {
            "sha256": _sha256(source),
            "size": len(source),
            "cleaned_sha256": _sha256(virtual),
            "cleaned_size": len(virtual),
        }
    full_hash = _content_hash(verified_files, cleaned=True)
    test_hash = _content_hash(
        {
            path: record
            for path, record in verified_files.items()
            if path == "data.yaml" or path.startswith("test/")
        },
        cleaned=True,
    )
    if full_hash != manifest.get("cleaning_view_hash"):
        raise ParityCleanError("complete cleaning view hash does not match manifest")
    if test_hash != manifest.get("cleaned_test_view_hash"):
        raise ParityCleanError("cleaned test view hash does not match manifest")
    return {"cleaning_view_hash": full_hash, "cleaned_test_view_hash": test_hash}


def write_audit_artifacts(output_dir: Path, audit: LabelAudit, manifest: Mapping[str, object]) -> tuple[Path, Path, Path]:
    """Write the requested JSON, CSV, and approved parity-clean manifest."""

    destination = Path(output_dir)
    try:
        protected = tuple(
            destination / name
            for name in (
                "invalid_label_audit.json",
                "invalid_label_audit.csv",
                "parity-clean-v1.json",
            )
        )
        existing = [path.name for path in protected if path.exists()]
        if existing:
            raise ParityCleanError(
                "refusing to overwrite existing parity audit artifact(s): {}".format(
                    ", ".join(existing)
                )
            )
        destination.mkdir(parents=True, exist_ok=True)
        json_path = destination / "invalid_label_audit.json"
        csv_path = destination / "invalid_label_audit.csv"
        manifest_path = destination / "parity-clean-v1.json"
        json_path.write_text(
            json.dumps(
                {
                    "protocol": "invalid-label-audit-v1",
                    "summary_by_split": audit.summary_by_split,
                    "split_counts": audit.split_counts,
                    "physical_invalid_rows": audit.physical_invalid_rows,
                    "issues": audit.issues,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        fields = ("split", "relative_path", "category", "line_number", "raw_content", "file_sha256")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in audit.issues:
                writer.writerow({name: row.get(name) for name in fields})
        manifest_path.write_text(json.dumps(dict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, TypeError, ValueError, csv.Error) as error:
        raise ParityCleanError("unable to write parity audit artifacts: {}".format(error)) from error
    return json_path, csv_path, manifest_path


def _find_split_names(root: Path) -> tuple[str, ...]:
    ordered = ("train", "valid", "val", "validation", "test")
    return tuple(name for name in ordered if (root / name / "images").is_dir() and (root / name / "labels").is_dir())


def _view_file_paths(root: Path, split_counts: Mapping[str, Mapping[str, object]]) -> tuple[Path, ...]:
    """Return only root data.yaml and image/label files consumed by parity."""

    paths = [root / "data.yaml"]
    for split in split_counts:
        for directory in (root / split / "images", root / split / "labels"):
            paths.extend(path for path in directory.rglob("*") if path.is_file())
    return tuple(sorted(paths))


def _issue(split: str, category: str, path: Path, root: Path, line_number: int | None = None, raw_content: str | None = None) -> dict[str, object]:
    return {
        "split": split,
        "relative_path": path.relative_to(root).as_posix(),
        "category": category,
        "line_number": line_number,
        "raw_content": raw_content,
        "file_sha256": _sha256(path.read_bytes()),
    }


def _audit_row(raw: str, class_count: int) -> dict[str, object]:
    fields = raw.strip().split()
    if len(fields) != 5:
        return {"categories": ("invalid_field_count",), "details": {"field_count": len(fields)}, "parsed": None}
    try:
        class_id = int(fields[0])
    except ValueError:
        return {"categories": ("invalid_class_id",), "details": {"parsed_class_id": None}, "parsed": None}
    try:
        values = tuple(float(value) for value in fields[1:])
    except ValueError:
        return {"categories": ("invalid_numeric",), "details": {"parsed_class_id": class_id}, "parsed": None}
    if not all(math.isfinite(value) for value in values):
        return {"categories": ("nan_or_inf",), "details": {"parsed_class_id": class_id, "coordinates": list(values)}, "parsed": None}
    x_center, y_center, width, height = values
    categories = []
    if class_id < 0 or class_id >= class_count:
        categories.append("invalid_class_id")
    if width <= 0.0:
        categories.append("width_lte_zero")
    if height <= 0.0:
        categories.append("height_lte_zero")
    if x_center < 0.0 or x_center > 1.0 or y_center < 0.0 or y_center > 1.0 or width > 1.0 or height > 1.0 or x_center - width / 2.0 < 0.0 or x_center + width / 2.0 > 1.0 or y_center - height / 2.0 < 0.0 or y_center + height / 2.0 > 1.0:
        categories.append("out_of_bounds")
    return {"categories": tuple(categories), "details": {"parsed_class_id": class_id, "coordinates": list(values)}, "parsed": (class_id, x_center, y_center, width, height)}


def _virtual_bytes(path: Path, removed_line_numbers: set[int]) -> bytes:
    if path.suffix.lower() != ".txt" or not removed_line_numbers:
        return path.read_bytes()
    return _canonical_label_bytes(
        tuple(
            line
            for line_number, line in enumerate(path.read_text(encoding="utf-8-sig", errors="surrogateescape").splitlines(), start=1)
            if line_number not in removed_line_numbers
        )
    )


def _canonical_label_bytes(lines: Sequence[str]) -> bytes:
    return ("\n".join(lines) + "\n").encode("utf-8") if lines else b""


def _content_hash(files: Mapping[str, Mapping[str, object]], *, cleaned: bool) -> str:
    digest = hashlib.sha256()
    for relative_path in sorted(files):
        record = files[relative_path]
        size_key = "cleaned_size" if cleaned else "size"
        hash_key = "cleaned_sha256" if cleaned else "sha256"
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record[size_key]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(record[hash_key]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


__all__ = [
    "LabelAudit",
    "ParityCleanError",
    "ParityCleanView",
    "audit_yolo_dataset",
    "build_parity_clean_manifest",
    "verify_parity_clean_view",
    "write_audit_artifacts",
]
