import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from perception.sam3.instances import InstanceSegmenter, select_instance, validate_instances
from perception.sam3.local_inference import LocalSAM3
from perception.sam3.remote_client import RemoteSAM3


class ArrayTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakePredictor:
    def __init__(self, overrides):
        self.overrides = overrides
        self.images = []
        self.resets = 0
        self.prompts = []
        self.results = []

    def reset_image(self):
        self.resets += 1

    def setup_model(self, verbose=False):
        self.model_loaded = True

    def set_image(self, image):
        self.images.append(image)

    def __call__(self, text):
        self.prompts.append(text)
        return self.results


@pytest.fixture
def local_setup(monkeypatch, tmp_path):
    import perception.config as config

    devices = []
    cache_releases = []
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True,
                                                empty_cache=lambda: cache_releases.append(True)))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "ultralytics.models", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "ultralytics.models.sam",
                        SimpleNamespace(SAM3SemanticPredictor=FakePredictor))
    monkeypatch.setitem(sys.modules, "clip", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "clip.simple_tokenizer", SimpleNamespace(SimpleTokenizer=lambda: object()))
    checks = SimpleNamespace(AUTOINSTALL=True, ONLINE=True)
    utils = SimpleNamespace(checks=checks, AUTOINSTALL=True, ONLINE=True)
    monkeypatch.setitem(sys.modules, "ultralytics.utils", utils)
    monkeypatch.setattr(sys.modules["ultralytics"], "utils", utils, raising=False)
    monkeypatch.setitem(sys.modules, "ultralytics.utils.events",
                        SimpleNamespace(events=SimpleNamespace(enabled=True)))
    monkeypatch.setenv("YOLO_AUTOINSTALL", "true")
    monkeypatch.setenv("YOLO_OFFLINE", "false")
    monkeypatch.setattr(config, "require_device", lambda module, device: devices.append(device))
    monkeypatch.setattr(config, "local_metadata", lambda name, checkpoint, device: {
        "backend": "local", "model": name, "device": device,
        "checkpoint_sha256": "test-digest",
    })
    (tmp_path / "sam3.pt").write_bytes(b"fake weights")
    image = tmp_path / "rectangular.png"
    Image.fromarray(np.zeros((6, 12, 3), np.uint8)).save(image)
    service = LocalSAM3(tmp_path)
    service._load()
    return service, image, devices, cache_releases


def result(masks, scores):
    return SimpleNamespace(masks=SimpleNamespace(data=ArrayTensor(masks)),
                           boxes=SimpleNamespace(conf=ArrayTensor(scores)))


def test_checkpoint_required_before_image_or_dependency_loading(tmp_path):
    service = LocalSAM3(tmp_path)
    assert service.device == "cuda"
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        service.segment(tmp_path / "missing.jpg")
    assert service.predictor is None


def test_explicit_device_and_metadata(local_setup):
    service, _, devices, _ = local_setup
    assert devices == ["cuda"]
    assert service.predictor.overrides["device"] == "cuda"
    assert service.predictor.model_loaded
    assert service.metadata["checkpoint_sha256"] == "test-digest"


def test_disables_dependency_autoinstall(local_setup):
    import os

    assert os.environ["YOLO_AUTOINSTALL"] == "false"
    assert not sys.modules["ultralytics.utils"].AUTOINSTALL
    assert not sys.modules["ultralytics.utils"].checks.AUTOINSTALL


def test_offline_before_import_and_disables_preexisting_analytics(local_setup, monkeypatch):
    import builtins
    import os

    service, _, _, _ = local_setup
    service.close()
    utils = sys.modules["ultralytics.utils"]
    events = sys.modules["ultralytics.utils.events"].events
    utils.ONLINE = utils.checks.ONLINE = events.enabled = True
    monkeypatch.setenv("YOLO_OFFLINE", "false")
    original_import = builtins.__import__
    inspected_imports = []

    def offline_import(name, *args, **kwargs):
        if name.startswith("ultralytics"):
            inspected_imports.append(name)
            assert os.environ["YOLO_OFFLINE"] == "true"
            assert os.environ["YOLO_AUTOINSTALL"] == "false"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", offline_import)
    service.load()
    assert inspected_imports
    assert not utils.ONLINE and not utils.checks.ONLINE
    assert not events.enabled


def test_missing_clip_fails_before_upstream_setup(local_setup, monkeypatch):
    service, _, _, _ = local_setup
    service.close()
    monkeypatch.setitem(sys.modules, "clip.simple_tokenizer", None)
    with pytest.raises(RuntimeError, match="CLIP tokenizer"):
        service.load()
    assert service.predictor is None


def test_missing_tokenizer_vocabulary_fails_before_upstream_setup(local_setup, monkeypatch):
    service, _, _, _ = local_setup
    service.close()

    def missing_vocabulary():
        raise FileNotFoundError("bpe.gz")

    monkeypatch.setattr(sys.modules["clip.simple_tokenizer"], "SimpleTokenizer", missing_vocabulary)
    with pytest.raises(RuntimeError, match="BPE vocabulary"):
        service.load()
    assert service.predictor is None


def test_device_failure_does_not_construct_model(local_setup, monkeypatch):
    import perception.config as config

    service, image, _, _ = local_setup
    service.close()

    def fail_device(module, device):
        raise RuntimeError("CUDA unavailable")

    monkeypatch.setattr(config, "require_device", fail_device)
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        service.segment(image)
    assert service.predictor is None


def test_load_failure_retains_no_partial_model(local_setup, monkeypatch):
    service, _, _, cache_releases = local_setup
    service.close()

    def failing_setup(self, verbose=False):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(FakePredictor, "setup_model", failing_setup)
    with pytest.raises(RuntimeError, match="out of memory while loading on cuda"):
        service.load()
    assert service.predictor is None
    assert len(cache_releases) == 2


def test_inference_oom_closes_model_without_changing_resolution(local_setup, monkeypatch):
    service, image, _, cache_releases = local_setup

    def failing_inference(self, text):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(FakePredictor, "__call__", failing_inference)
    with pytest.raises(RuntimeError, match="sufficient memory"):
        service.segment_instances(image, expected_shape=(6, 12))
    assert service.predictor is None
    assert service._image_key is None
    assert cache_releases == [True]


def test_instance_grid_scores_and_stage1_union(local_setup):
    service, image, _, _ = local_setup
    masks = np.zeros((2, 6, 12), dtype=bool)
    masks[0, 1:3, 1:4] = True
    masks[1, 4:6, 9:12] = True
    service.predictor.results = [result(masks, [0.9, 0.85])]
    instances = service.segment_instances(image, expected_shape=(6, 12))
    assert instances["scores"].dtype == np.float32
    np.testing.assert_array_equal(instances["masks"], masks)
    np.testing.assert_array_equal(service.segment(image), np.any(masks, axis=0))
    assert service.predictor.images == [str(image)]


@pytest.mark.parametrize("results", [[], [SimpleNamespace(masks=None)]])
def test_empty_detection_uses_original_resolution(local_setup, results):
    service, image, _, _ = local_setup
    service.predictor.results = results
    instances = service.segment_instances(image)
    assert instances["masks"].shape == (0, 6, 12)
    assert instances["scores"].shape == (0,)
    assert service.segment(image).shape == (6, 12)
    assert not service.segment(image).any()


def test_reject_square_or_letterboxed_masks_for_scale_fill_model(local_setup):
    service, image, _, _ = local_setup
    service.predictor.results = [result(np.zeros((1, 12, 12)), [0.9])]
    with pytest.raises(ValueError, match="invalid instance"):
        service.segment_instances(image)


def test_expected_shape_must_match_input_before_inference(local_setup):
    service, image, _, _ = local_setup
    with pytest.raises(ValueError, match="image shape"):
        service.segment_instances(image, expected_shape=(12, 12))
    assert service.predictor.images == []


def test_reuse_then_close_releases_embeddings_and_model(local_setup):
    service, image, devices, cache_releases = local_setup
    predictor = service.predictor
    service.segment(image, "yellow block")
    service.segment(image, "container")
    assert predictor.images == [str(image)]
    assert predictor.prompts == [["yellow block"], ["container"]]
    Image.fromarray(np.ones((6, 12, 3), np.uint8)).save(image)
    service.segment(image)
    assert len(predictor.images) == 2
    service.close()
    assert service.predictor is None and service._image_key is None
    assert cache_releases == [True]
    service.close()
    service.segment(image)
    assert service.predictor is not predictor
    assert devices == ["cuda", "cuda"]


def test_sim_selection_preserves_near_best_score_and_area_rule():
    masks = np.zeros((3, 6, 12), dtype=bool)
    masks[0, :, :6] = True
    masks[1, 1:3, 2:4] = True
    masks[2, 0, 0] = True
    instances = validate_instances(masks, [0.95, 0.93, 0.4], (6, 12))
    np.testing.assert_array_equal(select_instance(instances, "yellow block"), masks[1])
    np.testing.assert_array_equal(select_instance(instances, "container"), masks[0])


def test_instance_wrapper_empty_and_lifecycle(local_setup):
    service, image, _, cache_releases = local_setup
    segmenter = InstanceSegmenter(service)
    assert segmenter.segment(image, "block", (6, 12)).shape == (6, 12)
    assert segmenter.metadata is service.metadata
    segmenter.close()
    assert cache_releases == [True]


@pytest.mark.parametrize("scores", [None, [0.8, 0.9], [np.nan]])
def test_reject_invalid_instance_scores(scores):
    with pytest.raises(ValueError, match="instance"):
        validate_instances(np.zeros((1, 6, 12)), scores, (6, 12))


def test_remote_instances_original_shape_and_scoreless_union(monkeypatch, tmp_path):
    service = RemoteSAM3("unused", 0)
    masks = np.zeros((2, 6, 12), dtype=np.uint8)
    monkeypatch.setattr(service, "_request", lambda image, prompt: {"masks": masks, "scores": np.array([0.9, 0.8])})
    assert service.segment_instances(tmp_path / "unused", expected_shape=(6, 12))["masks"].shape == (2, 6, 12)
    with pytest.raises(ValueError, match="invalid mask resolution"):
        service.segment(tmp_path / "unused")
    monkeypatch.setattr(service, "_request", lambda image, prompt: {"masks": np.zeros((0, 288, 512), dtype=np.uint8)})
    assert not service.segment(tmp_path / "unused").any()
    assert service.segment_instances(tmp_path / "unused", expected_shape=(288, 512))["scores"].shape == (0,)


def test_remote_nonempty_instances_require_scores(monkeypatch, tmp_path):
    service = RemoteSAM3("unused", 0)
    monkeypatch.setattr(service, "_request", lambda image, prompt: {"masks": np.zeros((1, 6, 12), dtype=np.uint8)})
    with pytest.raises(ValueError, match="invalid instance"):
        service.segment_instances(tmp_path / "unused", expected_shape=(6, 12))
