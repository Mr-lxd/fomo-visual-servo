"""Read-only X-AnyLabeling annotation audit for collection specification 17.

Check training annotation JSONs without opening images. Legacy default-full
visibility and unambiguous containment are accepted with warnings; --strict
requires explicit visibility and group IDs for multi-target head/tail points.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


CLASSES = {"jellyfish", "fish", "tuna", "reflection tuna", "reflection jellyfish"}
VISIBILITY = {"full", "truncated", "occluded", "ignore"}
ORIENTATION = {"side", "oblique", "axial"}


def attribute(shape: dict, name: str, allowed: set[str]) -> str | None:
    value = (shape.get("attributes") or {}).get(name)
    description = (shape.get("description") or "").strip()
    return description if value is None and description in allowed else value


def inside(point: list[float], box: dict) -> bool:
    xs, ys = zip(*box["points"])
    return min(xs) - 2 <= point[0] <= max(xs) + 2 and min(ys) - 2 <= point[1] <= max(ys) + 2


def check(directory: Path, strict: bool = False) -> dict:
    problems, warnings, statistics = [], [], collections.Counter()
    complete_pairs = 0
    files = sorted(directory.glob("*.json"))
    annotation_files = 0
    for path in files:
        if path.name == "attributes.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        shapes = data["shapes"]
        width, height = data["imageWidth"], data["imageHeight"]
        annotation_files += 1
        boxes = [s for s in shapes if s["shape_type"] == "rectangle"]
        targets = [s for s in boxes if s["label"] in ("fish", "tuna")]
        points = [s for s in shapes if s["shape_type"] == "point"]
        assigned = collections.defaultdict(list)

        def problem(message: str) -> None:
            problems.append({"file": path.name, "message": message})

        def compatibility(message: str) -> None:
            (problems if strict else warnings).append({"file": path.name, "message": message})

        for shape in shapes:
            if shape["shape_type"] not in ("rectangle", "point"):
                problem(f"unexpected shape {shape['shape_type']} {shape['label']}")
        gids = [box.get("group_id") for box in boxes if box.get("group_id") is not None]
        if len(gids) != len(set(gids)):
            problem(f"duplicate box group_id {gids}")
        for point in points:
            statistics["points"] += 1
            label = point["label"]
            if label not in ("head", "tail"):
                problem(f"unknown point label {label}")
            if len(point["points"]) != 1:
                problem(f"{label} must contain exactly one coordinate")
                continue
            xy = point["points"][0]
            if not (0 <= xy[0] < width and 0 <= xy[1] < height):
                problem(f"{label} outside image bounds")
            state = attribute(point, "point_state", {"visible", "occluded"})
            if state not in (None, "visible", "occluded"):
                problem(f"invalid point_state {state}")
            if point.get("group_id") is not None:
                candidates = [box for box in targets if box.get("group_id") == point["group_id"]]
            else:
                candidates = [box for box in targets if inside(xy, box)]
                if len(targets) > 1 and len(candidates) == 1:
                    compatibility(f"{label} lacks multi-target group_id; uniquely resolved by containment")
            if len(candidates) != 1:
                problem(f"{label} gid={point.get('group_id')} matches {len(candidates)} fish/tuna boxes")
                continue
            assigned[id(candidates[0])].append(point)
        for box in boxes:
            label = box["label"]
            if label not in CLASSES:
                problem(f"unknown rectangle class {label}")
            visibility = attribute(box, "visibility", VISIBILITY)
            if visibility is None:
                visibility = "full"
                compatibility(f"{label} implicit visibility=full; no explicit attribute")
            if visibility not in VISIBILITY:
                problem(f"invalid {label} visibility {visibility}")
            statistics[f"boxes/{label}/{visibility}"] += 1
            bp = assigned[id(box)]
            heads = [p for p in bp if p["label"] == "head"]
            tails = [p for p in bp if p["label"] == "tail"]
            if len(heads) > 1 or len(tails) > 1:
                problem(f"{label} has {len(heads)} heads and {len(tails)} tails")
            if visibility == "ignore":
                if bp:
                    problem(f"ignore box has {len(bp)} points")
                continue
            if label not in ("fish", "tuna"):
                continue
            orientation = attribute(box, "orientation", ORIENTATION)
            if orientation not in ORIENTATION:
                problem(f"{label} {visibility} missing/invalid orientation {orientation}")
                continue
            if orientation == "axial" and bp:
                problem(f"axial box has {len(bp)} points")
            if orientation != "axial" and visibility == "full" and (len(heads) != 1 or len(tails) != 1):
                problem(f"full {orientation} {label} has {len(heads)}h/{len(tails)}t")
            for point in bp:
                if not inside(point["points"][0], box):
                    problem(f"{point['label']} outside its box")
            if orientation != "axial" and len(heads) == 1 and len(tails) == 1:
                complete_pairs += 1
    if not annotation_files:
        raise ValueError(f"no annotation JSON files in {directory}")
    return {"schema_version": "annotation-check-v1", "directory": str(directory.resolve()),
            "strict": strict, "files": annotation_files, "statistics": dict(sorted(statistics.items())),
            "complete_head_tail_pairs": complete_pairs, "problem_count": len(problems),
            "warning_count": len(warnings), "problems": problems, "warnings": warnings}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="one explicitly selected image/annotation directory")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = check(args.annotations, args.strict)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key not in ("warnings", "problems")}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for issue in report["problems"]:
        print(f"{issue['file']}: {issue['message']}")
    raise SystemExit(1 if report["problems"] else 0)


if __name__ == "__main__":
    main()
