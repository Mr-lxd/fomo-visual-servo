"""Session layout planning tests for the dataset capture tool."""

from __future__ import annotations

from datetime import date

import pytest

from fomo_servo.capture.session_layout import (
    derive_session_prefix,
    plan_next_session,
)


def test_first_session_gets_index_001(tmp_path) -> None:
    paths = plan_next_session(
        tmp_path / "datasets_raw" / "lab_pool", prefix="pool", date=date(2026, 8, 31)
    )

    assert paths.session_id == "pool-20260831-001"
    assert paths.session_dir.name == "pool-20260831-001"
    assert paths.session_dir.parent.name == "20260831"
    assert paths.frames_dir == paths.session_dir / "frames"
    assert paths.metadata_path == paths.session_dir / "metadata.json"
    assert paths.session_dir.is_dir()
    assert paths.frames_dir.is_dir()


def test_sessions_are_monotonic_within_one_day(tmp_path) -> None:
    output_root = tmp_path / "datasets_raw" / "lab_pool"

    first = plan_next_session(output_root, prefix="pool", date=date(2026, 8, 31))
    second = plan_next_session(output_root, prefix="pool", date=date(2026, 8, 31))
    third = plan_next_session(output_root, prefix="pool", date=date(2026, 8, 31))

    assert [first.session_id, second.session_id, third.session_id] == [
        "pool-20260831-001",
        "pool-20260831-002",
        "pool-20260831-003",
    ]


def test_new_day_restarts_numbering_and_never_touches_other_days(tmp_path) -> None:
    output_root = tmp_path / "datasets_raw" / "lab_pool"
    plan_next_session(output_root, prefix="pool", date=date(2026, 8, 31))
    plan_next_session(output_root, prefix="pool", date=date(2026, 8, 31))

    next_day = plan_next_session(output_root, prefix="pool", date=date(2026, 9, 1))

    assert next_day.session_id == "pool-20260901-001"
    assert next_day.session_dir.parent.name == "20260901"


def test_existing_session_is_never_overwritten_and_gaps_are_skipped(tmp_path) -> None:
    output_root = tmp_path / "datasets_raw" / "lab_pool"
    date_dir = output_root / "20260831"
    date_dir.mkdir(parents=True)
    (date_dir / "pool-20260831-001").mkdir()
    (date_dir / "pool-20260831-003").mkdir()

    planned = plan_next_session(output_root, prefix="pool", date=date(2026, 8, 31))

    assert planned.session_id == "pool-20260831-004"
    assert not (date_dir / "pool-20260831-002").exists()


@pytest.mark.parametrize("existing_artifact", ["metadata.json", "raw.avi", "frames"])
def test_partially_populated_session_directory_is_never_reused(
    tmp_path, existing_artifact
) -> None:
    output_root = tmp_path / "datasets_raw" / "lab_pool"
    existing_session = output_root / "20260831" / "pool-20260831-001"
    existing_session.mkdir(parents=True)
    artifact = existing_session / existing_artifact
    if existing_artifact == "frames":
        artifact.mkdir()
    else:
        artifact.write_bytes(b"existing data")

    planned = plan_next_session(output_root, prefix="pool", date=date(2026, 8, 31))

    assert planned.session_id == "pool-20260831-002"
    assert artifact.exists()


def test_file_collision_at_session_name_is_skipped(tmp_path) -> None:
    output_root = tmp_path / "datasets_raw" / "lab_pool"
    date_dir = output_root / "20260831"
    date_dir.mkdir(parents=True)
    collision = date_dir / "pool-20260831-001"
    collision.write_bytes(b"do not overwrite")

    planned = plan_next_session(output_root, prefix="pool", date=date(2026, 8, 31))

    assert planned.session_id == "pool-20260831-002"
    assert collision.read_bytes() == b"do not overwrite"


def test_derive_session_prefix_uses_output_root_leaf(tmp_path) -> None:
    assert derive_session_prefix(tmp_path / "datasets_raw" / "lab_pool") == "pool"
    assert derive_session_prefix(tmp_path / "datasets_raw" / "labpool") == "labpool"
    assert derive_session_prefix(tmp_path / "datasets_raw" / "field-river") == "field-river"
    assert (
        derive_session_prefix(tmp_path / "datasets_raw" / "___")
        == "capture"
    )
