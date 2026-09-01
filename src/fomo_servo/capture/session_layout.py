"""Deterministic capture session directory planning.

Layout contract::

    <output_root>/<YYYYMMDD>/<prefix>-<YYYYMMDD>-<NNN>/
        raw.avi            (first recording segment; raw_002.avi, ...)
        frames/
            frame_000001.jpg ...
        metadata.json

Session IDs are monotonic per output root and calendar day. Existing session
directories are never overwritten or reused; the planner always allocates the
next free index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date as _date_class
from pathlib import Path
from typing import Optional


_SESSION_INDEX_PATTERN = r"^{prefix}-{stamp}-(\d+)$"


@dataclass(frozen=True)
class SessionPaths:
    """Absolute filesystem locations for one capture session."""

    session_id: str
    session_dir: Path
    frames_dir: Path
    metadata_path: Path


def derive_session_prefix(output_root: Path) -> str:
    """Derive a session prefix from the output-root leaf name.

    ``datasets_raw/lab_pool`` yields ``pool`` (text after the last ``_``);
    overridable via the CLI ``--session-prefix``. Unusable names fall back to
    ``capture``.
    """

    leaf = Path(output_root).name.strip()
    candidate = leaf.rsplit("_", 1)[-1].strip()
    candidate = re.sub(r"[^A-Za-z0-9-]+", "-", candidate).strip("-").lower()
    return candidate or "capture"


def plan_next_session(
    output_root: Path, *, prefix: str, date: Optional[_date_class] = None
) -> SessionPaths:
    """Allocate and create the next session directory for ``date``.

    The planner scans the date directory for existing ``<prefix>-<stamp>-<NNN>``
    sessions, picks ``max(NNN) + 1``, and skips any already-existing directory
    name so existing data can never be reused or overwritten.
    """

    root = Path(output_root)
    day = date if date is not None else _date_class.today()
    stamp = day.strftime("%Y%m%d")
    date_dir = root / stamp
    date_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(
        _SESSION_INDEX_PATTERN.format(prefix=re.escape(prefix), stamp=stamp)
    )
    highest = 0
    for entry in date_dir.iterdir():
        match = pattern.match(entry.name)
        if entry.is_dir() and match:
            highest = max(highest, int(match.group(1)))

    index = highest + 1
    while True:
        session_dir = date_dir / "{}-{}-{:03d}".format(prefix, stamp, index)
        if not session_dir.exists():
            break
        index += 1

    frames_dir = session_dir / "frames"
    session_dir.mkdir(exist_ok=False)
    frames_dir.mkdir(exist_ok=False)
    return SessionPaths(
        session_id=session_dir.name,
        session_dir=session_dir,
        frames_dir=frames_dir,
        metadata_path=session_dir / "metadata.json",
    )
