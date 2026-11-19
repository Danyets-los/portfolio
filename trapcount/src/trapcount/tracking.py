from dataclasses import dataclass
from math import hypot

import numpy as np

from .core import Box, Detection, Object, assign, center, iou

METHODS = ("max-frame", "geometry", "appearance")


@dataclass(frozen=True)
class TrackerConfig:
    max_gap_seconds: float = 1.6
    center_gate: float = 2.2
    appearance_weight: float = 0.55
    min_observations: int = 1

    def __post_init__(self) -> None:
        if self.max_gap_seconds <= 0 or self.center_gate <= 0:
            raise ValueError("Tracking gap and center gate must be positive.")
        if not 0 <= self.appearance_weight <= 1 or self.min_observations < 1:
            raise ValueError("Invalid appearance weight or minimum observations.")


@dataclass
class Track:
    id: int
    last: Detection
    time: float
    velocity: tuple[float, float] = (0.0, 0.0)
    observations: int = 1

    def predict(self, time: float) -> Box:
        gap = min(time - self.time, 0.75)
        dx, dy = self.velocity[0] * gap, self.velocity[1] * gap
        left, top, right, bottom = self.last.box
        return left + dx, top + dy, right + dx, bottom + dy

    def update(self, detection: Detection, time: float) -> None:
        old_x, old_y = center(self.last.box)
        new_x, new_y = center(detection.box)
        elapsed = time - self.time
        self.velocity = (
            0.65 * (new_x - old_x) / elapsed + 0.35 * self.velocity[0],
            0.65 * (new_y - old_y) / elapsed + 0.35 * self.velocity[1],
        )
        feature = detection.appearance
        if feature is not None and self.last.appearance is not None:
            feature = 0.65 * self.last.appearance + 0.35 * feature
        self.last = Detection(detection.box, detection.score, feature)
        self.time = time
        self.observations += 1


@dataclass(frozen=True)
class Result:
    count: int
    frames: tuple[tuple[Object, ...], ...]
    track_ids: bool


def _association_cost(
    track: Track, detection: Detection, time: float, config: TrackerConfig, appearance: bool
) -> float | None:
    predicted = track.predict(time)
    px, py = center(predicted)
    x, y = center(detection.box)
    size = max(1.0, hypot(predicted[2] - predicted[0], predicted[3] - predicted[1]))
    displacement = hypot(x - px, y - py) / size
    if displacement > config.center_gate:
        return None
    geometry = 0.55 * (1 - iou(predicted, detection.box)) + 0.45 * (
        displacement / config.center_gate
    )
    if not appearance or track.last.appearance is None or detection.appearance is None:
        return geometry
    # Each channel histogram has unit mass before concatenation, as does their mean.
    visual = float(0.5 * np.abs(track.last.appearance - detection.appearance).sum())
    return (1 - config.appearance_weight) * geometry + config.appearance_weight * visual


def count(
    frames: list[tuple[Detection, ...]],
    timestamps: list[float],
    method: str,
    config: TrackerConfig,
) -> Result:
    if method not in METHODS:
        raise ValueError(f"Unknown tracking method: {method}")
    if len(frames) != len(timestamps) or any(a >= b for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("Frames require one timestamp each, in strictly increasing order.")
    if method == "max-frame":
        objects = tuple(
            tuple(Object(-1, det.box, det.score) for det in frame) for frame in frames
        )
        return Result(max(map(len, frames), default=0), objects, False)

    active: dict[int, Track] = {}
    all_tracks: dict[int, Track] = {}
    output: list[tuple[Object, ...]] = []
    next_id = 1
    for detections, time in zip(frames, timestamps):
        active = {
            key: track for key, track in active.items()
            if time - track.time <= config.max_gap_seconds + 1e-9
        }
        tracks = list(active.values())
        costs = [
            [_association_cost(track, det, time, config, method == "appearance") for det in detections]
            for track in tracks
        ]
        matched = dict(
            (col, tracks[row]) for row, col in assign(costs, unmatched_cost=0.72)
        )
        observations: list[Object] = []
        for index, detection in enumerate(detections):
            track = matched.get(index)
            if track is None:
                track = Track(next_id, detection, time)
                active[next_id] = all_tracks[next_id] = track
                next_id += 1
            else:
                track.update(detection, time)
            observations.append(Object(track.id, detection.box, detection.score))
        output.append(tuple(observations))
    accepted = {
        key for key, track in all_tracks.items() if track.observations >= config.min_observations
    }
    filtered = tuple(tuple(item for item in frame if item.identity in accepted) for frame in output)
    return Result(len(accepted), filtered, True)
