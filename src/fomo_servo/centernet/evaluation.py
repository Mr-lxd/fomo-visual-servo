"""Centroid-in-box matching, adjacency separation and size-usability metrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Optional, Sequence

import numpy as np

from .annotations import GroundTruthBox


@dataclass(frozen=True)
class Pred:
    """A thresholded prediction in original-image pixels; ``w/h`` only for CenterNet."""

    class_id: int
    score: float
    x: float
    y: float
    w: Optional[float] = None
    h: Optional[float] = None


@dataclass(frozen=True)
class ImageMatch:
    """Greedy one-to-one matches for one image.

    ``pairs`` are ``(pred_idx, gt_idx, distance)`` against the non-ignore GT list.
    ``false_positives`` excludes predictions whose centre lies inside an ``ignore`` box.
    """

    pairs: tuple[tuple[int, int, float], ...]
    false_positives: tuple[int, ...]
    false_negatives: tuple[int, ...]


def _inside(p: Pred, g: GroundTruthBox) -> bool:
    return g.x_min <= p.x <= g.x_max and g.y_min <= p.y <= g.y_max


def match_image(
    preds: Sequence[Pred], gts: Sequence[GroundTruthBox], *, class_agnostic: bool
) -> ImageMatch:
    """Same rule as ``CentroidEvaluator``: centroid inside the GT box, nearest pair first."""

    scored = [g for g in gts if g.visibility != "ignore"]
    ignored = [g for g in gts if g.visibility == "ignore"]
    candidates = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(scored):
            if not class_agnostic and p.class_id != g.class_id:
                continue
            if _inside(p, g):
                cx, cy = (g.x_min + g.x_max) / 2, (g.y_min + g.y_max) / 2
                candidates.append((hypot(p.x - cx, p.y - cy), pi, gi))
    candidates.sort()
    used_p, used_g, pairs = set(), set(), []
    for distance, pi, gi in candidates:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        pairs.append((pi, gi, distance))
    fps = tuple(
        pi for pi in range(len(preds))
        if pi not in used_p and not any(_inside(preds[pi], g) for g in ignored)
    )
    fns = tuple(gi for gi in range(len(scored)) if gi not in used_g)
    return ImageMatch(tuple(pairs), fps, fns)


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def adjacent_pairs(gts: Sequence[GroundTruthBox], max_distance: float) -> list[tuple[int, int]]:
    """Index pairs (into the non-ignore GT list) whose box centres are within ``max_distance``."""

    scored = [g for g in gts if g.visibility != "ignore"]
    centres = [((g.x_min + g.x_max) / 2, (g.y_min + g.y_max) / 2) for g in scored]
    return [
        (i, j)
        for i in range(len(scored))
        for j in range(i + 1, len(scored))
        if hypot(centres[i][0] - centres[j][0], centres[i][1] - centres[j][1]) <= max_distance
    ]


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rho with average ranks for ties (no SciPy dependency)."""

    def ranks(values: Sequence[float]) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        order = np.argsort(values, kind="mergesort")
        out = np.empty(len(values))
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
                j += 1
            out[order[i : j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return out

    if len(a) < 3:
        return float("nan")
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate_threshold(
    preds_per_image: Sequence[Sequence[Pred]],
    gts_per_image: Sequence[Sequence[GroundTruthBox]],
    *,
    class_agnostic: bool,
    adjacent_distance: float,
) -> dict:
    """Aggregate TP/FP/FN, centre error, adjacency separation and per-visibility recall."""

    tp = fp = fn = 0
    distances: list[float] = []
    adjacent_total = adjacent_both = 0
    by_vis: dict[str, list[int]] = {}
    for preds, gts in zip(preds_per_image, gts_per_image):
        m = match_image(preds, gts, class_agnostic=class_agnostic)
        scored = [g for g in gts if g.visibility != "ignore"]
        tp += len(m.pairs)
        fp += len(m.false_positives)
        fn += len(m.false_negatives)
        distances += [d for _, _, d in m.pairs]
        matched = {gi for _, gi, _ in m.pairs}
        for gi, g in enumerate(scored):
            hit_total = by_vis.setdefault(g.visibility, [0, 0])
            hit_total[1] += 1
            hit_total[0] += gi in matched
        for i, j in adjacent_pairs(gts, adjacent_distance):
            adjacent_total += 1
            adjacent_both += (i in matched) and (j in matched)
    precision, recall, f1 = prf(tp, fp, fn)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "center_error_median_px": float(np.median(distances)) if distances else float("nan"),
        "adjacent_pairs": adjacent_total,
        "adjacent_both_detected": adjacent_both,
        "adjacent_separation_rate": adjacent_both / adjacent_total if adjacent_total else float("nan"),
        "recall_by_visibility": {k: {"hit": v[0], "total": v[1]} for k, v in by_vis.items()},
    }
