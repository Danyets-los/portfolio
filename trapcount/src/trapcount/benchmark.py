from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .core import Clip, Detection, Object, assign, iou, sample_indices
from .detectors import Detector, read_image, with_appearance
from .tracking import METHODS, Result, TrackerConfig, count


@dataclass(frozen=True)
class Frame:
    detections: tuple[Detection, ...]
    reference: tuple[Detection, ...]
    detector_ms: float
    feature_ms: float
    reference_feature_ms: float
    decode_ms: float
    gray: bool


@dataclass(frozen=True)
class Example:
    clip: Clip
    fps: float
    indices: list[int]
    result: Result
    metrics: dict
    baseline: int


def ids(frames: tuple[tuple[Object, ...], ...]) -> set[int]:
    return {obj.identity for frame in frames for obj in frame}


def identity_metrics(truth: tuple, result: Result, threshold: float = 0.5) -> dict:
    potential: Counter = Counter()
    matched_truth: set[int] = set()
    truth_tracks: dict[int, set[int]] = defaultdict(set)
    predicted_tracks: dict[int, set[int]] = defaultdict(set)
    frame_matches = 0
    for reference, predicted in zip(truth, result.frames):
        scores = [[iou(gt.box, item.box) for item in predicted] for gt in reference]
        for gt, row in zip(reference, scores):
            for item, overlap in zip(predicted, row):
                if overlap >= threshold:
                    potential[gt.identity, item.identity] += 1
        costs = [[1 - overlap if overlap >= threshold else None for overlap in row] for row in scores]
        pairs = assign(costs, unmatched_cost=len(reference) + 1.0)
        frame_matches += len(pairs)
        for left, right in pairs:
            gt_id, pred_id = reference[left].identity, predicted[right].identity
            matched_truth.add(gt_id)
            truth_tracks[gt_id].add(pred_id)
            predicted_tracks[pred_id].add(gt_id)

    gt_total = sum(map(len, truth))
    pred_total = sum(map(len, result.frames))
    metrics = {
        "spatial_matches": frame_matches,
        "visible_boxes": gt_total,
        "predicted_boxes": pred_total,
        "spatially_recalled_animals": len(matched_truth),
        "fragmented_ids": None,
        "merged_tracks": None,
        "idf1": None,
        "idtp": None,
    }
    if result.track_ids:
        reference_ids, prediction_ids = sorted(ids(truth)), sorted(ids(result.frames))
        costs = [
            [-potential[gt, pred] if potential[gt, pred] else None for pred in prediction_ids]
            for gt in reference_ids
        ]
        idtp = sum(potential[reference_ids[a], prediction_ids[b]] for a, b in assign(costs, 0.0))
        metrics.update(
            fragmented_ids=sum(len(tracks) > 1 for tracks in truth_tracks.values()),
            merged_tracks=sum(len(tracks) > 1 for tracks in predicted_tracks.values()),
            idtp=idtp,
            idf1=2 * idtp / (gt_total + pred_total) if gt_total + pred_total else None,
        )
    return metrics


def clip_metrics(clip: Clip, indices: list[int], result: Result) -> dict:
    sampled = tuple(clip.truth[index] for index in indices)
    reference = len(ids(clip.truth))
    visible = len(ids(sampled))
    error = result.count - reference
    return {
        "reference_count": reference,
        "visible_reference_count": visible,
        "sampling_misses": reference - visible,
        "prediction": result.count,
        "signed_error": error,
        "absolute_error": abs(error),
        **identity_metrics(sampled, result),
    }


def _prepare_frame(clip: Clip, index: int, detector: Detector) -> Frame:
    start = perf_counter()
    image = read_image(clip.images[index])
    decoded = perf_counter()
    predictions = detector.detect(image, clip.images[index])
    detected = perf_counter()
    detections = tuple(with_appearance(image, det) for det in predictions)
    embedded = perf_counter()
    reference = tuple(
        with_appearance(image, Detection(item.box)) for item in clip.truth[index]
    )
    reference_embedded = perf_counter()
    thumbnail = np.asarray(image.resize((32, 24)), dtype=np.int16)
    gray = float(np.ptp(thumbnail, axis=2).mean()) <= 12
    return Frame(
        detections, reference,
        1000 * (detected - decoded), 1000 * (embedded - detected),
        1000 * (reference_embedded - embedded), 1000 * (decoded - start), gray,
    )


def evaluate(
    clips: list[Clip],
    detector: Detector,
    rates: tuple[float, ...],
    config: TrackerConfig,
    example_count: int,
    progress,
) -> tuple[list[dict], list[Example]]:
    if not clips:
        raise ValueError("Evaluation received no clips.")
    if example_count < 0:
        raise ValueError("Number of examples cannot be negative.")
    rows = []
    examples: list[Example] = []
    first_path = clips[0].images[0]
    detector.detect(read_image(first_path), first_path)
    for position, clip in enumerate(clips, start=1):
        requested_frames = {rate: sample_indices(len(clip.images), clip.fps, rate) for rate in rates}
        prepared = {
            index: _prepare_frame(clip, index, detector)
            for index in sorted(set().union(*requested_frames.values()))
        }
        tags = list(clip.slices)
        if sum(frame.gray for frame in prepared.values()) > len(prepared) / 2:
            tags.append("low-color")
        for fps, indices in requested_frames.items():
            batch = [prepared[index] for index in indices]
            timestamps = [index / clip.fps for index in indices]
            decode_ms = sum(frame.decode_ms for frame in batch)
            baseline = 0
            for source in ("detector", "oracle"):
                detector_ms = sum(frame.detector_ms for frame in batch) if source == "detector" else 0
                for method in METHODS:
                    frames = [frame.detections if source == "detector" else frame.reference for frame in batch]
                    feature_ms = 0.0
                    if method == "appearance":
                        feature_ms = sum(
                            frame.feature_ms if source == "detector" else frame.reference_feature_ms
                            for frame in batch
                        )
                    start = perf_counter()
                    result = count(frames, timestamps, method, config)
                    tracking_ms = 1000 * (perf_counter() - start)
                    metrics = {
                        "clip": clip.id,
                        "site": clip.site,
                        "split": clip.split,
                        "slices": "|".join(tags),
                        "source": source,
                        "method": method,
                        "fps": fps,
                        "sampled_frames": len(indices),
                        "decode_ms": decode_ms,
                        "detector_ms": detector_ms,
                        "feature_ms": feature_ms,
                        "tracking_ms": tracking_ms,
                        "work_ms": decode_ms + detector_ms + feature_ms + tracking_ms,
                        **clip_metrics(clip, indices, result),
                    }
                    rows.append(metrics)
                    if source == "detector" and method == "max-frame":
                        baseline = result.count
                    if source == "detector" and result.track_ids and example_count:
                        candidate = Example(clip, fps, indices, result, metrics, baseline)
                        examples.append(candidate)
        progress(position, len(clips), clip.id)
    examples.sort(key=_example_rank, reverse=True)
    return rows, _select_examples(examples, example_count)


def _example_rank(example: Example) -> tuple:
    reference = example.metrics["reference_count"]
    return (
        max(abs(example.baseline - reference), example.metrics["absolute_error"]),
        abs(example.baseline - example.result.count),
        example.metrics["sampling_misses"],
        len(example.clip.slices),
    )


def _select_examples(examples: list[Example], limit: int) -> list[Example]:
    selected: list[Example] = []
    for criterion in ("improvement", "regression", "sampling", "all"):
        for example in examples:
            if len(selected) == limit:
                return selected
            if any(item.clip.id == example.clip.id for item in selected):
                continue
            before = abs(example.baseline - example.metrics["reference_count"])
            after = example.metrics["absolute_error"]
            if (
                (criterion == "improvement" and after < before)
                or (criterion == "regression" and after > before)
                or (criterion == "sampling" and example.metrics["sampling_misses"] > 0)
                or criterion == "all"
            ):
                selected.append(example)
                if criterion != "all":
                    break
    return selected


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def _summarize(rows: list[dict]) -> dict:
    count_with_ids = [row for row in rows if row["idtp"] is not None]
    id_total = sum(row["visible_boxes"] + row["predicted_boxes"] for row in count_with_ids)
    return {
        "clips": len(rows),
        "sites": len({row["site"] for row in rows}),
        "mae": _mean(rows, "absolute_error"),
        "bias": _mean(rows, "signed_error"),
        "under_rate": sum(row["signed_error"] < 0 for row in rows) / len(rows),
        "over_rate": sum(row["signed_error"] > 0 for row in rows) / len(rows),
        "exact_rate": sum(row["signed_error"] == 0 for row in rows) / len(rows),
        "mean_sampling_misses": _mean(rows, "sampling_misses"),
        "idf1": 2 * sum(row["idtp"] for row in count_with_ids) / id_total if id_total else None,
        "mean_ms": _mean(rows, "work_ms"),
        "p95_ms": float(np.percentile([row["work_ms"] for row in rows], 95)),
    }


def _paired_interval(pairs: list[tuple[str, int]], seed: int, repeats: int) -> dict:
    by_site: dict[str, list[int]] = defaultdict(list)
    for site, delta in pairs:
        by_site[site].append(delta)
    point = float(np.mean([delta for _, delta in pairs]))
    result = {"delta_mae": point, "low": None, "high": None, "sites": len(by_site)}
    if len(by_site) >= 5 and repeats:
        groups = list(by_site.values())
        sums = np.asarray([sum(group) for group in groups])
        sizes = np.asarray([len(group) for group in groups])
        rng = np.random.default_rng(seed)
        choices = rng.integers(0, len(groups), size=(repeats, len(groups)))
        scores = sums[choices].sum(axis=1) / sizes[choices].sum(axis=1)
        result.update(low=float(np.percentile(scores, 2.5)), high=float(np.percentile(scores, 97.5)))
    return result


def summarize(rows: list[dict], seed: int, bootstrap: int) -> tuple[list[dict], list[dict], list[dict]]:
    grouped = defaultdict(list)
    sliced = defaultdict(list)
    for row in rows:
        key = (row["source"], row["method"], row["fps"])
        grouped[key].append(row)
        for label in filter(None, row["slices"].split("|")):
            sliced[(*key, label)].append(row)
    summary = []
    intervals = []
    slices = []
    for (source, method, fps), group in sorted(grouped.items()):
        summary.append({"source": source, "method": method, "fps": fps, **_summarize(group)})
        if method == "max-frame":
            continue
        baselines = {
            row["clip"]: row["absolute_error"] for row in grouped[(source, "max-frame", fps)]
        }
        pairs = [(row["site"], row["absolute_error"] - baselines[row["clip"]]) for row in group]
        intervals.append({
            "source": source, "method": method, "fps": fps,
            **_paired_interval(pairs, seed, bootstrap),
        })
    for (source, method, fps, label), group in sorted(sliced.items()):
        slices.append({
            "source": source, "method": method, "fps": fps, "slice": label, **_summarize(group),
        })
    return summary, intervals, slices
