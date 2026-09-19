"""Deterministic, non-overwriting capture session layout."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SessionPaths:
    session_id: str
    session_dir: Path
    frames_dir: Path
    metadata_path: Path
    frame_index_path: Path


def sanitize_prefix(value: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9-]+", "-", value.strip()).strip("-").lower()
    return candidate or "capture"


def plan_next_session(
    output_root: Path,
    *,
    prefix: str = "capture",
    date: Optional[Date] = None,
) -> SessionPaths:
    """Create the next <prefix>-<YYYYMMDD>-<NNN> directory without reuse."""

    root = Path(output_root)
    day = date if date is not None else Date.today()
    stamp = day.strftime("%Y%m%d")
    safe_prefix = sanitize_prefix(prefix)
    day_dir = root / stamp
    day_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(
        rf"^{re.escape(safe_prefix)}-{stamp}-(\d+)$"
    )
    highest = 0
    for entry in day_dir.iterdir():
        match = pattern.match(entry.name)
        if entry.is_dir() and match:
            highest = max(highest, int(match.group(1)))

    index = highest + 1
    while True:
        session_id = f"{safe_prefix}-{stamp}-{index:03d}"
        session_dir = day_dir / session_id
        if not session_dir.exists():
            break
        index += 1

    frames_dir = session_dir / "frames"
    session_dir.mkdir(exist_ok=False)
    frames_dir.mkdir(exist_ok=False)
    return SessionPaths(
        session_id=session_id,
        session_dir=session_dir,
        frames_dir=frames_dir,
        metadata_path=session_dir / "metadata.json",
        frame_index_path=session_dir / "frame_index.csv",
    )
