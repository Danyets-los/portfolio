import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from math import ceil, isfinite
from pathlib import Path
from time import perf_counter

from . import __version__
from .benchmark import evaluate, summarize
from .core import sample_indices
from .dataset import annotation_path, audit, check_disjoint, fetch, load_clips, select_clips, write_manifest
from .detectors import CachedMegaDetector, FixtureDetector, MegaDetector, read_image, with_appearance
from .fixture import generate
from .media import decoded_frame, image_sequence, render_preview, video_frames
from .report import write_benchmark, write_inference, write_json
from .tracking import METHODS, TrackerConfig, count


def _rate(raw: str) -> float:
    value = float(raw)
    if not isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Expected a positive, finite frame rate.")
    return value


def _positive(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return value


def _probability(raw: str) -> float:
    value = float(raw)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("Expected a number between zero and one.")
    return value


def _dataset_options(parser, annotations=True) -> None:
    parser.add_argument("--root", type=Path, default=Path("data/sa-fari"))
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--limit", type=_positive, help="Seeded site-diverse sample. Default: all clips.")
    parser.add_argument("--seed", type=int, default=42)
    if annotations:
        parser.add_argument("--annotations", type=Path, help="Use a local extended annotation file.")


def _detector_options(parser) -> None:
    parser.add_argument("--detector", choices=("megadetector", "cached", "fixture"), default="megadetector")
    parser.add_argument("--model", default="MDV5A", help="MegaDetector model alias or trusted local weights path.")
    parser.add_argument("--threshold", type=_probability, default=0.30)
    parser.add_argument("--image-size", type=_positive)
    parser.add_argument("--cpu", action="store_true", help="Force MegaDetector to CPU.")
    parser.add_argument("--predictions", type=Path, help="Standard MegaDetector JSON for the cached backend.")
    parser.add_argument("--prediction-root", type=Path, help="Directory relative to which JSON paths were exported.")


def _tracking_options(parser) -> None:
    parser.add_argument("--gap", type=_rate, default=1.6, help="Maximum idle time in seconds.")
    parser.add_argument("--center-gate", type=_rate, default=2.2)
    parser.add_argument("--appearance-weight", type=_probability, default=0.55)
    parser.add_argument("--min-observations", type=_positive, default=1)


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trapcount", description="Camera-trap counting and sampling benchmark.")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    fixture = commands.add_parser("fixture", help="Generate a small annotated dataset for offline checks.")
    fixture.add_argument("--root", type=Path, default=Path("data/fixture"))
    fixture.add_argument("--sites", type=_positive, default=4, help="Number of locations per split.")

    download = commands.add_parser("fetch", help="Download SA-FARI annotations and selected frame sequences.")
    _dataset_options(download, annotations=False)
    download.add_argument("--workers", type=_positive, default=8)
    download.add_argument("--revision", default="main", help="Hugging Face branch or exact revision.")

    inspect = commands.add_parser("audit", help="Audit annotation coverage, location splits and frame files.")
    _dataset_options(inspect)
    inspect.add_argument("--decode", action="store_true", help="Decode every selected frame to validate it.")

    benchmark = commands.add_parser("benchmark", help="Compare count methods at multiple frame rates.")
    _dataset_options(benchmark)
    _detector_options(benchmark)
    _tracking_options(benchmark)
    benchmark.add_argument("--fps", type=_rate, nargs="+", default=(6, 3, 1))
    benchmark.add_argument("--bootstrap", type=int, default=2000)
    benchmark.add_argument("--examples", type=int, default=6)
    benchmark.add_argument("--out", type=Path, required=True)

    infer = commands.add_parser("infer", help="Count animals in a video or directory of ordered frames.")
    _detector_options(infer)
    _tracking_options(infer)
    infer.add_argument("--input", type=Path, required=True)
    infer.add_argument("--source-fps", type=_rate, default=6, help="Capture FPS when input is a directory.")
    infer.add_argument("--fps", type=_rate, default=3)
    infer.add_argument("--method", choices=METHODS, default="appearance")
    infer.add_argument("--preview-frames", type=_positive, default=90)
    infer.add_argument("--out", type=Path, required=True)
    return parser


def _detector(args):
    if args.detector == "fixture":
        return FixtureDetector()
    if args.detector == "cached":
        if args.predictions is None or args.prediction_root is None:
            raise ValueError("The cached detector requires --predictions and --prediction-root.")
        return CachedMegaDetector(args.predictions, args.prediction_root, args.threshold)
    return MegaDetector(args.model, args.threshold, args.cpu, args.image_size)


def _tracking(args) -> TrackerConfig:
    return TrackerConfig(args.gap, args.center_gate, args.appearance_weight, args.min_observations)


def _progress(position: int, total: int, clip: str) -> None:
    if position == 1 or position % 10 == 0 or position == total:
        print(f"[{position}/{total}] {clip}", file=sys.stderr, flush=True)


def _separate_locations(root: Path, split: str, clips) -> bool:
    other = "train" if split == "test" else "test"
    if annotation_path(root, other).is_file():
        check_disjoint(clips, load_clips(root, other))
        return True
    return False


def _versions() -> dict:
    packages = {}
    for library in ("numpy", "Pillow", "megadetector", "torch", "opencv-python"):
        try:
            packages[library] = version(library)
        except PackageNotFoundError:
            continue
    return packages


def _benchmark(args) -> None:
    if args.examples < 0 or args.bootstrap < 0 or args.seed < 0:
        raise ValueError("Examples, bootstrap repeats and seed must be nonnegative.")
    annotations = args.annotations or annotation_path(args.root, args.split)
    all_clips = load_clips(args.root, args.split, annotations)
    disjoint = _separate_locations(args.root, args.split, all_clips)
    clips = select_clips(all_clips, args.limit, args.seed)
    inventory = audit(clips)
    if inventory["missing"]:
        raise ValueError(
            f"{inventory['missing']} frame(s) missing. Run fetch with matching split, limit and seed. "
            f"First missing: {inventory['examples'][0]}"
        )
    if clips and args.detector == "fixture" and clips[0].source != "synthetic fixture":
        raise ValueError("The fixture detector is only for generated fixture benchmarks.")
    rates = tuple(sorted(set(args.fps), reverse=True))
    if not clips or rates[0] > min(clip.fps for clip in clips):
        raise ValueError("Benchmark requires clips and rates between zero and the annotated FPS.")
    detector = _detector(args)
    config = _tracking(args)
    rows, examples = evaluate(clips, detector, rates, config, args.examples, _progress)
    stats, paired, slices = summarize(rows, args.seed, args.bootstrap)
    with annotations.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    metadata = {
        "trapcount": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": clips[0].source, "split": args.split, "annotations_sha256": digest,
        "location_overlap_checked": disjoint, "detector": detector.name,
        "detector_threshold": args.threshold, "detector_image_size": args.image_size,
        "device": detector.device, "detector_mode": args.detector,
        "fps": rates, "seed": args.seed, "bootstrap": args.bootstrap,
        "clip_ids": [clip.id for clip in clips], "tracker": asdict(config),
        "timing": "decode + detector call + optional appearance extraction + association; one warmup",
        "platform": platform.platform(), "python": platform.python_version(),
        "packages": _versions(),
    }
    write_manifest(args.out / "manifest.csv", clips, args.root)
    write_benchmark(args.out, metadata, rows, stats, paired, slices, examples)
    print(args.out / "index.html")


def _input_frames(path: Path, source_fps: float, target_fps: float):
    if path.is_dir():
        frames = image_sequence(path)
        for index in sample_indices(len(frames), source_fps, target_fps):
            yield index / source_fps, frames[index], read_image(frames[index])
        return
    if not path.is_file():
        raise ValueError(f"Input does not exist: {path}")
    tick = 0
    for index, fps, capture in video_frames(path):
        if target_fps > fps:
            raise ValueError(f"Target FPS exceeds the video's source rate ({fps:.2f}).")
        if index >= ceil(tick * fps / target_fps - 1e-9):
            yield index / fps, path, decoded_frame(capture)
            tick += 1


def _infer(args) -> None:
    if args.input.is_file() and args.detector == "cached":
        raise ValueError("The cached detector expects a directory of frames.")
    config = _tracking(args)
    detector = _detector(args)
    frames, timestamps, preview = [], [], []
    start = perf_counter()
    for timestamp, path, image in _input_frames(args.input, args.source_fps, args.fps):
        detections = detector.detect(image, path)
        if args.method == "appearance":
            detections = tuple(with_appearance(image, det) for det in detections)
        frames.append(detections)
        timestamps.append(timestamp)
        if len(preview) < args.preview_frames:
            preview.append(image)
    if not frames:
        raise ValueError("Input contains no decodable frames.")
    result = count(frames, timestamps, args.method, config)
    duration = 1000 * (perf_counter() - start)
    args.out.mkdir(parents=True, exist_ok=True)
    preview_result = replace(result, frames=result.frames[:len(preview)])
    render_preview(preview, timestamps[:len(preview)], preview_result, args.out / "tracks.gif")
    write_inference(args.out, {
        "input": args.input.name, "detector": detector.name, "device": detector.device,
        "method": args.method, "fps": args.fps, "tracker": asdict(config),
        "detector_threshold": args.threshold, "detector_image_size": args.image_size,
        "count": result.count, "has_track_ids": result.track_ids,
        "sampled_frames": len(frames), "preview_frames": len(preview), "elapsed_ms": duration,
        "frames": [
            {
                "seconds": timestamp,
                "objects": [
                    {"track_id": obj.identity if result.track_ids else None, "xyxy": obj.box, "score": obj.score}
                    for obj in observations
                ],
            }
            for timestamp, observations in zip(timestamps, result.frames)
        ],
    })
    print(f"Count: {result.count} | {len(frames)} frames | {args.out / 'index.html'}")


def main(argv=None) -> int:
    parser = _cli()
    args = parser.parse_args(argv)
    try:
        if args.command == "fixture":
            generate(args.root, args.sites)
            print(args.root)
        elif args.command == "fetch":
            clips = fetch(args.root, args.split, args.limit, args.seed, args.workers, args.revision)
            manifest = args.root / f"fetch-{args.split}-{args.limit or 'all'}-{args.seed}.csv"
            write_manifest(manifest, clips, args.root)
            print(manifest)
        elif args.command == "audit":
            clips = load_clips(args.root, args.split, args.annotations)
            disjoint = _separate_locations(args.root, args.split, clips)
            summary = audit(select_clips(clips, args.limit, args.seed), decode=args.decode)
            summary["location_overlap_checked"] = disjoint
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return int(bool(summary["missing"] or summary["unreadable"] or not summary["clips"]))
        elif args.command == "benchmark":
            _benchmark(args)
        elif args.command == "infer":
            _infer(args)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"trapcount: {error}", file=sys.stderr)
        return 1
    return 0
