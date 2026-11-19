import itertools
import json
import random
import tarfile
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from trapcount.benchmark import clip_metrics, identity_metrics, summarize
from trapcount.cli import main
from trapcount.core import Clip, Detection, Object, assign, sample_indices
from trapcount.dataset import (
    UNKNOWN_LOCATION, _AnnotationJSON, check_disjoint, extract_frames, load_clips, relative_path, select_clips,
)
from trapcount.detectors import CachedMegaDetector
from trapcount.tracking import Result, TrackerConfig, count

BOX = (0, 0, 10, 10)
OTHER = (12, 0, 22, 10)


class TrackingTests(unittest.TestCase):
    def test_sample_grid(self):
        self.assertEqual(sample_indices(8, 6, 3), [0, 2, 4, 6])
        self.assertEqual(sample_indices(8, 6, 1), [0, 6])
        self.assertEqual(sample_indices(12, 6, 4), [0, 2, 3, 5, 6, 8, 9, 11])
        with self.assertRaises(ValueError):
            sample_indices(8, 3, 6)

    def test_assignment_with_forbidden_edges_matches_brute_force(self):
        rng = random.Random(12)
        for rows, columns in itertools.product(range(1, 5), repeat=2):
            for _ in range(12):
                costs = [[rng.choice([None, 0.1, 0.4, 0.8, 2.0]) for _ in range(columns)] for _ in range(rows)]
                chosen = assign(costs, unmatched_cost=1.0)
                actual = sum(costs[a][b] for a, b in chosen) + rows - len(chosen)
                expected = min(
                    sum(1.0 if b >= columns else costs[a][b] for a, b in enumerate(sequence))
                    for sequence in itertools.permutations(range(columns + rows), rows)
                    if all(b >= columns or costs[a][b] is not None for a, b in enumerate(sequence))
                )
                self.assertAlmostEqual(actual, expected)

    def test_appearance_preserves_identity_on_crossing(self):
        first = np.array([1.0, 0.0])
        second = np.array([0.0, 1.0])
        frames = [
            (Detection(BOX, appearance=first), Detection(OTHER, appearance=second)),
            (Detection(BOX, appearance=second), Detection(OTHER, appearance=first)),
        ]
        geometry = count(frames, [0, 1], "geometry", TrackerConfig())
        appearance = count(frames, [0, 1], "appearance", TrackerConfig())
        self.assertEqual([obj.identity for obj in geometry.frames[1]], [1, 2])
        self.assertEqual([obj.identity for obj in appearance.frames[1]], [2, 1])
        self.assertEqual(appearance.count, 2)

    def test_timestamp_gap_and_single_observation(self):
        data = [(Detection(BOX),), (), (Detection(BOX),)]
        self.assertEqual(count(data, [0, 1, 2], "geometry", TrackerConfig()).count, 2)
        self.assertEqual(count(data, [0, 0.7, 1.4], "geometry", TrackerConfig()).count, 1)
        config = TrackerConfig(min_observations=2)
        self.assertEqual(count(data, [0, 1, 2], "geometry", config).frames, ((), (), ()))

    def test_idf1_detects_fragmentation_despite_perfect_boxes(self):
        truth = ((Object(1, BOX),), (Object(1, BOX),))
        result = Result(2, ((Object(1, BOX),), (Object(2, BOX),)), True)
        measured = identity_metrics(truth, result)
        self.assertEqual(measured["idf1"], 0.5)
        self.assertEqual(measured["fragmented_ids"], 1)
        self.assertEqual(measured["spatial_matches"], 2)

    def test_count_target_includes_objects_missing_from_selected_frames(self):
        box1, box2 = Object(1, BOX), Object(2, OTHER)
        truth = ((box1,), (box1, box2), (box1,))
        clip = Clip("short", "site", "test", "test", 6.0, (), truth)
        result = Result(1, ((box1,), (box1,)), True)
        measured = clip_metrics(clip, [0, 2], result)
        self.assertEqual(measured["visible_reference_count"], 1)
        self.assertEqual(measured["sampling_misses"], 1)
        self.assertEqual(measured["absolute_error"], 1)
        self.assertEqual(measured["idf1"], 1)


class DataTests(unittest.TestCase):
    def test_extract_frames_copies_only_requested_image_members(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "frames.tar.gz"
            source = BytesIO()
            Image.new("RGB", (4, 3), "red").save(source, format="JPEG")
            jpeg = source.getvalue()
            with tarfile.open(archive, "w:gz") as bundle:
                for name in ("../outside.jpg", "clip/extra.jpg", "./clip/001.jpg"):
                    member = tarfile.TarInfo(name)
                    member.size = len(jpeg)
                    bundle.addfile(member, BytesIO(jpeg))
            base = root / "frames"
            wanted = base / "clip" / "001.jpg"
            extract_frames(archive, base, [wanted])
            self.assertEqual(list(base.rglob("*.jpg")), [wanted])
            self.assertEqual(wanted.read_bytes(), jpeg)
            self.assertFalse((root / "outside.jpg").exists())

    def test_streaming_annotation_reader_normalizes_missing_metadata_across_chunks(self):
        key = b'"video_creation_datetime": '
        prefix = b'{"note":"NaN","spacing":"'
        for offset in (1, len(key) - 1, len(key) + 1):
            filler = b" " * (128 * 1024 - len(prefix) - len(b'",') - offset)
            payload = prefix + filler + b'",' + key + b'NaN,"Kingdom":NaN,"location_id":NaN}'
            with _AnnotationJSON(BytesIO(payload)) as source:
                result = json.loads(source.read(), parse_constant=self.fail)
            self.assertEqual(result["note"], "NaN")
            self.assertIsNone(result["video_creation_datetime"])
            self.assertIsNone(result["Kingdom"])
            self.assertIsNone(result["location_id"])

    def test_unknown_camera_locations_do_not_imply_train_test_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            annotations = root / "labels.json"
            annotations.write_text(json.dumps({
                "videos": [{"id": 4, "video_name": "clip", "location_id": float("nan"), "file_names": ["1.jpg"]}],
                "annotations": [],
            }), encoding="utf-8")
            train = load_clips(root, "train", annotations)
            test = load_clips(root, "test", annotations)
            self.assertEqual(test[0].site, UNKNOWN_LOCATION)
            check_disjoint(train, test)

    def test_dataset_layout_and_site_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            annotations = root / "labels.json"
            annotations.write_text(json.dumps({
                "videos": [{
                    "id": 4, "video_name": "clip", "location_id": "site",
                    "file_names": ["001.jpg", "sa_fari_test/JPEGImages_6fps/clip/002.jpg"],
                }],
                "annotations": [{"video_id": 4, "bboxes": [[0, 0, 10, 10], None]}],
            }), encoding="utf-8")
            clips = load_clips(root, "test", annotations)
            self.assertEqual(clips[0].images[1], root / "frames/test/clip/002.jpg")
            self.assertEqual(clips[0].truth[0][0].box, BOX)
            self.assertEqual(clips[0].truth[1], ())
            with self.assertRaises(ValueError):
                check_disjoint(clips, clips)
            with self.assertRaises(ValueError):
                relative_path("clips/../escape.jpg")

    def test_subset_is_reproducible_and_site_diverse(self):
        clips = [Clip(str(i), str(i // 3), "test", "test", 6, (), ()) for i in range(15)]
        selected = select_clips(clips, 5, 42)
        self.assertEqual(len({clip.site for clip in selected}), 5)
        self.assertEqual(selected, select_clips(clips[::-1], 5, 42))

    def test_cached_detector_filters_people_and_uses_xywh(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = root / "md.json"
            payload.write_text(json.dumps({"images": [{"file": "clip/frame.jpg", "detections": [
                {"category": "1", "bbox": [0.1, 0.2, 0.3, 0.4], "conf": 0.8},
                {"category": "2", "bbox": [0, 0, 1, 1], "conf": 0.9},
            ]}]}), encoding="utf-8")
            detector = CachedMegaDetector(payload, root, 0.3)
            results = detector.detect(Image.new("RGB", (100, 50)), root / "clip/frame.jpg")
            self.assertEqual(len(results), 1)
            np.testing.assert_allclose(results[0].box, (10, 10, 40, 30))
            with self.assertRaises(ValueError):
                detector.detect(Image.new("RGB", (100, 50)), root / "missing.jpg")

    def test_offline_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, result = root / "dataset", root / "run"
            self.assertEqual(main(["fixture", "--root", str(data), "--sites", "1"]), 0)
            self.assertEqual(main([
                "benchmark", "--root", str(data), "--detector", "fixture",
                "--out", str(result), "--examples", "1", "--limit", "2",
            ]), 0)
            stored = json.loads((result / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["metadata"]["dataset"], "synthetic fixture")
            self.assertEqual(len(stored["summary"]), 18)
            self.assertTrue((result / "examples/clip-01.gif").is_file())
            for row in stored["paired"]:
                self.assertIsNone(row["low"])
            self.assertEqual(main([
                "infer", "--input", str(data / "frames/test/site-01-sequential"),
                "--detector", "fixture", "--out", str(root / "demo"),
            ]), 0)
            predictions = json.loads((root / "demo/tracks.json").read_text(encoding="utf-8"))
            self.assertEqual(predictions["count"], 2)
            self.assertEqual(predictions["sampled_frames"], 21)


if __name__ == "__main__":
    unittest.main()
