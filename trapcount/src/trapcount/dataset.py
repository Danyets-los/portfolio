import csv
import json
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from math import isfinite
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import urlopen

from PIL import Image

from .core import Clip, Object, iou

SA_FARI = "facebook/SA-FARI"
GCS = "https://storage.googleapis.com/cxl-public-camera-trap/sa_fari"


def annotation_path(root: Path, split: str) -> Path:
    return root / "annotation" / f"sa_fari_{split}_ext.json"


def relative_path(raw: str) -> str:
    path = PurePosixPath(raw)
    if not path.parts or path.is_absolute() or ".." in path.parts or any(
        char in raw for char in ("\\", ":", "\x00")
    ):
        raise ValueError(f"Unsafe dataset frame path: {raw!r}")
    return path.as_posix()


def _frame_name(raw: str, video_name: str) -> str:
    path = relative_path(raw)
    parts = path.split("/")
    if "JPEGImages_6fps" in parts:
        path = relative_path("/".join(parts[parts.index("JPEGImages_6fps") + 1:]))
    if "/" not in path:
        path = relative_path(video_name) + "/" + path
    return path


def _records(annotations: Path):
    """Read annotation rows without holding the dense masks for a split in memory."""
    if annotations.stat().st_size >= 32 * 1024 * 1024:
        try:
            from ijson import items
        except ImportError as error:
            raise RuntimeError("Install trapcount[real] to stream large annotation files.") from error
        with annotations.open("rb") as file:
            info = next(items(file, "info", use_float=True), {})
            file.seek(0)
            videos = list(items(file, "videos.item", use_float=True))
            file.seek(0)
            for track in items(file, "annotations.item", use_float=True):
                yield "track", {key: track[key] for key in ("video_id", "bboxes", "iscrowd") if key in track}
        yield "videos", videos
        yield "info", info
    else:
        with annotations.open(encoding="utf-8") as file:
            raw = json.load(file)
        yield "videos", raw["videos"]
        yield "info", raw.get("info", {})
        for track in raw["annotations"]:
            yield "track", track


def load_clips(root: Path, split: str, annotations: Path | None = None) -> list[Clip]:
    annotations = annotations or annotation_path(root, split)
    by_video = defaultdict(list)
    videos = {}
    source = "SA-FARI"
    for kind, record in _records(annotations):
        if kind == "track":
            by_video[str(record["video_id"])].append(record)
        elif kind == "videos":
            videos = {str(video["id"]): video for video in record}
            if len(videos) != len(record):
                raise ValueError("Duplicate video IDs in annotations.")
        elif record.get("synthetic"):
            source = "synthetic fixture"
    if extras := by_video.keys() - videos.keys():
        raise ValueError(f"Track references an unknown video: {min(extras)}")

    clips: list[Clip] = []
    for key, video in videos.items():
        location = video.get("location_id")
        if location is None or location == "":
            raise ValueError(f"Missing location_id in {key}. Use the *_ext.json annotation file.")
        name = str(video["video_name"])
        names = [_frame_name(value, name) for value in video["file_names"]]
        if not names or len(names) != len(set(names)):
            raise ValueError(f"Missing or duplicate frames in {name}.")
        paths = tuple(root / "frames" / split / path for path in names)
        frame_labels: list[list[Object]] = [[] for _ in paths]
        for track_id, annotation in enumerate(by_video[key], start=1):
            if annotation.get("iscrowd"):
                continue
            boxes = annotation["bboxes"]
            if len(boxes) != len(paths):
                raise ValueError(f"{name}: track length differs from annotated frames.")
            for index, box in enumerate(boxes):
                if box is None:
                    continue
                x, y, width, height = map(float, box)
                if not all(isfinite(value) for value in (x, y, width, height)):
                    raise ValueError(f"{name}: invalid box at frame {index}.")
                if width > 0 and height > 0:
                    frame_labels[index].append(Object(track_id, (x, y, x + width, y + height)))

        truth = tuple(tuple(frame) for frame in frame_labels)
        tracks = Counter(obj.identity for frame in truth for obj in frame)
        slices = ["group" if len(tracks) > 1 else "single" if tracks else "empty"]
        if tracks and min(tracks.values()) <= 3:
            slices.append("brief")
        if any(iou(a.box, b.box) >= 0.15 for frame in truth for a, b in combinations(frame, 2)):
            slices.append("overlap")
        slices.extend(video.get("fixture_slices", []))
        clips.append(Clip(key, str(location), split, source, 6, paths, truth, tuple(slices)))
    return clips


def select_clips(clips: list[Clip], limit: int | None, seed: int) -> list[Clip]:
    if limit is None or limit >= len(clips):
        return sorted(clips, key=lambda clip: clip.id)
    if limit <= 0:
        raise ValueError("Limit must be positive.")
    # Round-robin across sites, with seeded random order within each site.
    by_site = defaultdict(list)
    for clip in sorted(clips, key=lambda clip: clip.id):
        by_site[clip.site].append(clip)
    rng = random.Random(seed)
    sites = sorted(by_site)
    rng.shuffle(sites)
    for group in by_site.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < limit:
        for site in sites:
            if by_site[site]:
                selected.append(by_site[site].pop())
                if len(selected) == limit:
                    break
    return selected


def check_disjoint(train: list[Clip], test: list[Clip]) -> None:
    shared = {clip.site for clip in train} & {clip.site for clip in test}
    if shared:
        raise ValueError(f"Training and evaluation share {len(shared)} location(s).")


def audit(clips: list[Clip], decode: bool = False) -> dict:
    missing = []
    unreadable = []
    for clip in clips:
        for path in clip.images:
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
            elif decode:
                try:
                    with Image.open(path) as image:
                        image.load()
                except (OSError, ValueError):
                    unreadable.append(str(path))
    return {
        "clips": len(clips),
        "locations": len({clip.site for clip in clips}),
        "frames": sum(len(clip.images) for clip in clips),
        "slices": dict(sorted(Counter(tag for clip in clips for tag in clip.slices).items())),
        "missing": len(missing),
        "unreadable": len(unreadable),
        "examples": (missing + unreadable)[:12],
    }


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with urlopen(url, timeout=30) as response, tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".download-", delete=False
        ) as file:
            temporary = Path(file.name)
            shutil.copyfileobj(response, file)
        if not temporary.stat().st_size:
            raise OSError(f"Empty response for {url}")
        with Image.open(temporary) as image:
            image.verify()
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch(root: Path, split: str, limit: int | None, seed: int, workers: int, revision: str) -> list[Clip]:
    path = annotation_path(root, split)
    if not path.is_file():
        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import GatedRepoError
        except ImportError as error:
            raise RuntimeError("Install trapcount[real] to download SA-FARI annotations.") from error
        try:
            downloaded = hf_hub_download(
                repo_id=SA_FARI, filename=f"annotation/{path.name}", repo_type="dataset",
                revision=revision, cache_dir=root / ".hf-cache",
            )
        except GatedRepoError as error:
            raise RuntimeError(
                "SA-FARI annotations require dataset access. Open "
                "https://huggingface.co/datasets/facebook/SA-FARI in your account, "
                "accept the dataset terms and authenticate with 'hf auth login'."
            ) from error
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.part")
        shutil.copyfile(downloaded, temporary)
        temporary.replace(path)

    clips = select_clips(load_clips(root, split), limit, seed)
    base = root / "frames" / split
    files = sorted({frame for clip in clips for frame in clip.images})

    def download_frame(frame: Path) -> None:
        if frame.is_file() and frame.stat().st_size:
            return
        relative = quote(frame.relative_to(base).as_posix(), safe="/")
        url = f"{GCS}/sa_fari_{split}/JPEGImages_6fps/{relative}"
        _download(url, frame)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(download_frame, files):
            pass
    return clips


def write_manifest(path: Path, clips: list[Clip], root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("clip", "location", "split", "source", "frames", "reference_count", "slices", "first_frame"))
        for clip in clips:
            identities = {obj.identity for frame in clip.truth for obj in frame}
            writer.writerow((
                clip.id, clip.site, clip.split, clip.source, len(clip.images),
                len(identities), "|".join(clip.slices), clip.images[0].relative_to(root).as_posix(),
            ))
