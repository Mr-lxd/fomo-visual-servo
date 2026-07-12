"""Small dependency-free connected-component extraction for FOMO heatmaps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


class ConnectedComponentsError(ValueError):
    """Raised when a heatmap mask has an invalid shape or connectivity."""


@dataclass(frozen=True)
class ConnectedComponent:
    """One deterministic component represented by ``(grid_x, grid_y)`` cells."""

    cells: Tuple[Tuple[int, int], ...]

    @property
    def area(self) -> int:
        """Return the number of heatmap cells in this component."""

        return len(self.cells)


def find_connected_components(
    mask: np.ndarray, *, connectivity: int = 8
) -> Tuple[ConnectedComponent, ...]:
    """Find deterministic 4- or 8-neighbor components in a boolean ``[G,G]`` mask."""

    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ConnectedComponentsError("mask must have shape [G,G]")
    if connectivity not in {4, 8}:
        raise ConnectedComponentsError("connectivity must be 4 or 8")
    boolean_mask = mask.astype(bool, copy=False)
    height, width = boolean_mask.shape
    visited = np.zeros_like(boolean_mask, dtype=bool)
    if connectivity == 4:
        neighbors = ((-1, 0), (0, -1), (0, 1), (1, 0))
    else:
        neighbors = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )

    components = []
    for grid_y in range(height):
        for grid_x in range(width):
            if not boolean_mask[grid_y, grid_x] or visited[grid_y, grid_x]:
                continue
            stack = [(grid_x, grid_y)]
            visited[grid_y, grid_x] = True
            cells = []
            while stack:
                current_x, current_y = stack.pop()
                cells.append((current_x, current_y))
                for offset_y, offset_x in neighbors:
                    next_x = current_x + offset_x
                    next_y = current_y + offset_y
                    if (
                        0 <= next_x < width
                        and 0 <= next_y < height
                        and boolean_mask[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        stack.append((next_x, next_y))
            components.append(ConnectedComponent(tuple(sorted(cells, key=lambda cell: (cell[1], cell[0])))))
    return tuple(components)
