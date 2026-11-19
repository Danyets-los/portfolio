import json
from math import isfinite
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, ImageOps

from .core import Box, Detection

MARKERS = ((247, 184, 95), (90, 224, 211), (236, 114, 158))


def read_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


class Detector(Protocol):
    name: str

    def detect(self, image: Image.Image, frame: Path) -> tuple[Detection, ...]: ...


def box_from_xywh(values: list[float], size: tuple[int, int], normalized: bool) -> Box | None:
    x, y, width, height = map(float, values)
    if not all(isfinite(value) for value in (x, y, width, height)):
        return None
    columns, rows = size
    sx, sy = (columns, rows) if normalized else (1, 1)
    box = (
        max(0.0, x * sx), max(0.0, y * sy),
        min(float(columns), (x + width) * sx), min(float(rows), (y + height) * sy),
    )
    return box if box[0] < box[2] and box[1] < box[3] else None


def with_appearance(image: Image.Image, detection: Detection) -> Detection:
    """L1-normalized RGB histograms from the central portion of the box."""
    x0, y0, x1, y1 = detection.box
    margin_x, margin_y = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
    bounds = (x0 + margin_x, y0 + margin_y, x1 - margin_x, y1 - margin_y)
    crop = np.asarray(image.crop(bounds).resize((16, 16), Image.Resampling.BILINEAR))
    bins = [np.bincount(crop[:, :, channel].ravel() // 32, minlength=8) for channel in range(3)]
    return Detection(detection.box, detection.score, np.concatenate(bins) / crop.size)


def _from_md(records: list[dict], size: tuple[int, int], threshold: float) -> tuple[Detection, ...]:
    detections = []
    for row in records:
        score = float(row["conf"])
        if str(row["category"]) != "1" or not isfinite(score) or score < threshold:
            continue
        box = box_from_xywh(row["bbox"], size, normalized=True)
        if box is not None:
            detections.append(Detection(box, score))
    return tuple(sorted(detections, key=lambda det: (-det.score, det.box)))


class MegaDetector:
    def __init__(self, model: str, threshold: float, cpu: bool, image_size: int | None) -> None:
        try:
            from megadetector.detection.run_detector import load_detector
        except ImportError as error:
            raise RuntimeError("Install trapcount[real] to run MegaDetector.") from error
        self.model = load_detector(model, force_cpu=cpu)
        self.name = f"MegaDetector / {model}"
        self.device = str(getattr(self.model, "device", "cpu" if cpu else "auto"))
        self.threshold = threshold
        self.image_size = image_size

    def detect(self, image: Image.Image, frame: Path) -> tuple[Detection, ...]:
        result = self.model.generate_detections_one_image(
            image, image_id=frame.name, detection_threshold=self.threshold, image_size=self.image_size,
        )
        if result.get("failure") or "detections" not in result:
            raise RuntimeError(f"MegaDetector failed on {frame}: {result.get('failure', 'missing detections')}")
        return _from_md(result["detections"], image.size, self.threshold)


class CachedMegaDetector:
    """Read standard MegaDetector JSON with paths relative to --prediction-root."""

    def __init__(self, file: Path, root: Path, threshold: float) -> None:
        with file.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        self.records = {row["file"].replace("\\", "/"): row for row in raw["images"]}
        if len(self.records) != len(raw["images"]):
            raise ValueError("Duplicate frame paths in MegaDetector JSON.")
        categories = raw.get("detection_categories", {})
        if categories and categories.get("1") != "animal":
            raise ValueError("Expected category 1 to be animal in MegaDetector JSON.")
        self.root = root.resolve()
        self.threshold = threshold
        self.name = f"Detection JSON / {file.name}"
        self.device = "cached"

    def detect(self, image: Image.Image, frame: Path) -> tuple[Detection, ...]:
        key = frame.resolve().relative_to(self.root).as_posix()
        row = self.records.get(key)
        if row is None or row.get("failure") or row.get("detections") is None:
            raise ValueError(f"Missing usable MegaDetector record for {key}")
        return _from_md(row["detections"], image.size, self.threshold)


class FixtureDetector:
    """Detect colored markers by pixels, using no ground-truth labels."""

    name = "Synthetic color-marker detector"
    device = "cpu"

    def detect(self, image: Image.Image, frame: Path) -> tuple[Detection, ...]:
        pixels = np.asarray(image, dtype=np.int16)
        detections = []
        for color in MARKERS:
            mask = (np.abs(pixels - color) <= 12).all(axis=2)
            ys, xs = np.nonzero(mask)
            if xs.size >= 30:
                box = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
                detections.append(Detection(box, score=0.99))
        return tuple(detections)
