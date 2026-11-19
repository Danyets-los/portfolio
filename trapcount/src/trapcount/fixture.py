import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .dataset import annotation_path
from .detectors import MARKERS

SCENES = ("sequential", "occlusion", "crossing", "brief-pass", "false-trigger", "single-pass")
DIMENSIONS = (480, 260)
LENGTH = 42


def _background(site: int, night: bool) -> Image.Image:
    width, height = DIMENSIONS
    rng = np.random.default_rng(site + 179)
    gradient = np.linspace(0, 1, height)[:, None, None]
    sky = np.array((16, 31, 44) if night else (41, 76, 79))
    earth = np.array((35, 42, 50) if night else (70, 82, 63))
    pixels = sky + gradient * (earth - sky) + rng.normal(0, 2.5, (height, width, 1))
    image = Image.fromarray(np.clip(pixels, 0, 255).astype("uint8"))
    draw = ImageDraw.Draw(image)
    for _ in range(22):
        x, y = int(rng.integers(0, width)), int(rng.integers(30, 115))
        draw.polygon([(x - 34, 170), (x, y), (x + 34, 170)], fill=(26, 50, 51) if not night else (24, 42, 53))
    draw.polygon([(0, 166), (width, 160), (width, 207), (0, 235)], fill=(74, 78, 66) if not night else (45, 49, 54))
    for _ in range(92):
        x, y = int(rng.integers(0, width)), int(rng.integers(167, height))
        draw.line((x, y, x + 3, y - 3), fill=(96, 97, 77) if not night else (59, 62, 67))
    return image


def _paths(scene: str, frame: int, speed: float) -> dict[int, tuple[int, int]]:
    i = frame
    if scene in {"single-pass", "false-trigger"}:
        return {0: (int(40 + i * speed), 177)} if scene == "single-pass" else {}
    if scene == "sequential":
        return {0: (int(32 + 8 * i), 176)} if i < 19 else (
            {1: (int(28 + 8 * (i - 22)), 178)} if i >= 22 else {}
        )
    if scene == "occlusion":
        paths = {0: (int(35 + 8.7 * i), 174), 1: (int(400 - 6.4 * i), 204)}
        if 18 <= i <= 23:
            paths.pop(0)
        return paths
    if scene == "crossing":
        return {0: (int(35 + 8.8 * i), 175), 1: (int(412 - 8.5 * i), 185)}
    extras = {1: (335, 158)} if 9 <= i <= 10 else {}
    return {0: (int(36 + i * speed), 184), **extras}


def _draw_marker(draw: ImageDraw.ImageDraw, x: int, y: int, identity: int) -> list[int]:
    draw.ellipse((x - 17, y + 12, x + 30, y + 20), fill=(25, 35, 38))
    color = MARKERS[identity]
    draw.ellipse((x - 17, y - 9, x + 18, y + 9), fill=color)
    draw.ellipse((x + 12, y - 16, x + 25, y - 1), fill=color)
    draw.line((x - 10, y + 6, x - 15, y + 16), fill=color, width=4)
    draw.line((x + 10, y + 6, x + 16, y + 17), fill=color, width=4)
    draw.ellipse((x + 20, y - 12, x + 22, y - 10), fill=(30, 37, 45))
    return [x - 17, y - 16, 43, 35]


def generate(root: Path, sites_per_split: int = 4) -> None:
    if sites_per_split < 1:
        raise ValueError("Generate at least one site per split.")
    for split_index, split in enumerate(("train", "test")):
        videos = []
        annotations = []
        for site_index in range(sites_per_split):
            site = split_index * sites_per_split + site_index
            night = site_index % 2 == 1
            background = _background(site, night)
            for scene in SCENES:
                name = f"site-{site:02d}-{scene}"
                path = root / "frames" / split / name
                path.mkdir(parents=True, exist_ok=True)
                boxes: dict[int, list] = {}
                names = []
                for index in range(LENGTH):
                    image = background.copy()
                    draw = ImageDraw.Draw(image)
                    if scene == "occlusion":
                        draw.rectangle((205, 64, 239, 193), fill=(30, 42, 49))
                    for identity, (x, y) in _paths(scene, index, 3 + 0.5 * site_index).items():
                        bbox = _draw_marker(draw, x, y, identity)
                        boxes.setdefault(identity, [None] * LENGTH)[index] = bbox
                    if scene == "false-trigger" and 28 <= index <= 34:
                        draw.ellipse((310, 156, 323, 164), fill=MARKERS[2])
                    filename = f"{index:04d}.jpg"
                    image.save(path / filename, quality=94, subsampling=0)
                    names.append(f"{name}/{filename}")
                videos.append({
                    "id": name, "video_name": name, "location_id": f"fixture-{site}",
                    "file_names": names, "fixture_slices": [scene, "night-fixture" if night else "day-fixture"],
                })
                annotations.extend(
                    {"id": len(annotations) + 1, "video_id": name, "bboxes": bbox}
                    for bbox in boxes.values()
                )
        path = annotation_path(root, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "info": {"synthetic": True}, "videos": videos, "annotations": annotations,
        }), encoding="utf-8")
