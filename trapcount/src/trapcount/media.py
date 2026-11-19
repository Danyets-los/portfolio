import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .core import Object, center
from .detectors import read_image
from .tracking import Result

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
COLORS = ("#82e6ce", "#ffcf84", "#fa99b2", "#a7c9ff", "#c1e785", "#d7b9f0")


def _sort_key(path: Path) -> tuple:
    return tuple(
        (0, int(token)) if token.isdigit() else (1, token.casefold())
        for token in re.split(r"(\d+)", path.name)
    )


def image_sequence(path: Path) -> tuple[Path, ...]:
    return tuple(sorted(
        (item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES),
        key=_sort_key,
    ))


def video_frames(path: Path):
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Install trapcount[real] for video decoding, or pass a frame directory.") from error
    video = cv2.VideoCapture(str(path))
    try:
        if not video.isOpened():
            raise ValueError(f"Could not open video: {path}")
        fps = video.get(cv2.CAP_PROP_FPS)
        if not 0 < fps < 1000:
            raise ValueError(f"Video does not report usable FPS: {path}")
        index = 0
        while video.grab():
            yield index, fps, video
            index += 1
    finally:
        video.release()


def decoded_frame(capture) -> Image.Image:
    ok, pixels = capture.retrieve()
    if not ok:
        raise ValueError("Could not decode the selected video frame.")
    return Image.fromarray(pixels[:, :, ::-1].copy())


def render_preview(
    frames: list[Path | Image.Image],
    timestamps: list[float],
    result: Result,
    output: Path,
    labels: tuple[tuple[Object, ...], ...] | None = None,
) -> None:
    if not frames or len(frames) != len(timestamps) or len(frames) != len(result.frames):
        raise ValueError("A preview needs one image, timestamp and prediction per frame.")
    previews = []
    trails: dict[int, list[tuple[float, float]]] = {}
    font = ImageFont.load_default(size=13)
    title_font = ImageFont.load_default(size=15)
    for index, item in enumerate(frames):
        image = read_image(item) if isinstance(item, Path) else item.copy()
        width, height = image.size
        image.thumbnail((640, 420), Image.Resampling.LANCZOS)
        sx, sy = image.width / width, image.height / height
        draw = ImageDraw.Draw(image)
        if labels is not None:
            for obj in labels[index]:
                x0, y0, x1, y1 = obj.box
                draw.rectangle((x0 * sx, y0 * sy, x1 * sx, y1 * sy), outline="#eaf0f2", width=1)
        for obj in result.frames[index]:
            x0, y0, x1, y1 = obj.box
            box = (x0 * sx, y0 * sy, x1 * sx, y1 * sy)
            color = COLORS[obj.identity % len(COLORS)] if result.track_ids else COLORS[0]
            if result.track_ids:
                x, y = center(box)
                path = trails.setdefault(obj.identity, [])
                path.append((x, y))
                if len(path) > 1:
                    draw.line(path[-16:], fill=color, width=2)
            draw.rectangle(box, outline=color, width=2)
            tag = f"ID {obj.identity:02d}" if result.track_ids else f"{obj.score:.2f}"
            tag_top = max(0, box[1] - 17)
            draw.rectangle((box[0], tag_top, box[0] + 42, tag_top + 16), fill=color)
            draw.text((box[0] + 3, tag_top), tag, fill="#0c1a20", font=font)
        draw.rectangle((0, 0, image.width, 26), fill="#12212b")
        status = f"{timestamps[index]:05.2f}s  |  in frame {len(result.frames[index])}  |  clip estimate {result.count}"
        draw.text((10, 4), status, fill="#e7efed", font=title_font)
        previews.append(image)
    output.parent.mkdir(parents=True, exist_ok=True)
    intervals = [max(20, round((second - first) * 1000)) for first, second in zip(timestamps, timestamps[1:])]
    duration = intervals + [intervals[-1] if intervals else 700]
    previews[0].save(
        output, save_all=True, append_images=previews[1:],
        duration=duration, loop=0, disposal=2, optimize=False,
    )
