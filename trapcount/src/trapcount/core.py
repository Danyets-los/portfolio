from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Sequence

import numpy as np

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    box: Box
    score: float = 1.0
    appearance: np.ndarray | None = None


@dataclass(frozen=True)
class Object:
    identity: int
    box: Box
    score: float = 1.0


@dataclass(frozen=True)
class Clip:
    id: str
    site: str
    split: str
    source: str
    fps: float
    images: tuple[Path, ...]
    truth: tuple[tuple[Object, ...], ...]
    slices: tuple[str, ...] = ()


def sample_indices(length: int, source_fps: float, target_fps: float) -> list[int]:
    """Select the first frame at or after each sampling tick."""
    if not 0 < target_fps <= source_fps:
        raise ValueError("Target FPS must be positive and at most source FPS.")
    ticks = ceil(length * target_fps / source_fps)
    return [
        frame
        for tick in range(ticks)
        if (frame := ceil(tick * source_fps / target_fps - 1e-9)) < length
    ]


def center(box: Box) -> tuple[float, float]:
    left, top, right, bottom = box
    return (left + right) / 2, (top + bottom) / 2


def iou(first: Box, second: Box) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    overlap = width * height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - overlap
    return overlap / union if union > 0 else 0.0


def _assignment(cost: Sequence[Sequence[float]]) -> list[tuple[int, int]]:
    """Rectangular Hungarian algorithm for finite costs and rows <= columns."""
    rows = len(cost)
    if not rows:
        return []
    columns = len(cost[0])
    u, v = [0.0] * (rows + 1), [0.0] * (columns + 1)
    matched = [0] * (columns + 1)
    previous = [0] * (columns + 1)
    for row in range(1, rows + 1):
        matched[0] = row
        column = 0
        distance = [float("inf")] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[column] = True
            source = matched[column]
            delta, next_column = float("inf"), 0
            for candidate in range(1, columns + 1):
                if used[candidate]:
                    continue
                reduced = cost[source - 1][candidate - 1] - u[source] - v[candidate]
                if reduced < distance[candidate]:
                    distance[candidate], previous[candidate] = reduced, column
                if distance[candidate] < delta:
                    delta, next_column = distance[candidate], candidate
            for candidate in range(columns + 1):
                if used[candidate]:
                    u[matched[candidate]] += delta
                    v[candidate] -= delta
                else:
                    distance[candidate] -= delta
            column = next_column
            if matched[column] == 0:
                break
        while column:
            parent = previous[column]
            matched[column] = matched[parent]
            column = parent
    return [(row - 1, col - 1) for col, row in enumerate(matched) if col and row]


def assign(cost: Sequence[Sequence[float | None]], unmatched_cost: float) -> list[tuple[int, int]]:
    """Match one-to-one, allowing each row to stay unmatched at a fixed cost."""
    if not cost or not cost[0]:
        return []
    columns = len(cost[0])
    # Each row has an available dummy column, so forbidden edges are never selected.
    padded = [
        [value if value is not None else unmatched_cost + 1 for value in row]
        + [unmatched_cost] * len(cost)
        for row in cost
    ]
    return [
        (row, col)
        for row, col in sorted(_assignment(padded))
        if col < columns and cost[row][col] is not None
    ]
