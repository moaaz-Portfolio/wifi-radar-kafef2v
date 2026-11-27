"""End-to-end check of the deep-learning stack. Needs torch (skipped without it).   python -m pytest tests/test_torch_stack.py"""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from athena import data, learning as L  # noqa: E402
from athena.live import Predictor  # noqa: E402
from athena.model import ActivityNet, ActivityNetV2, build_model  # noqa: E402
from athena.train import augment, train  # noqa: E402
from test_crossval import LABELS, make_dataset  # noqa: E402


def test_model_shapes_and_gradients():
    x = torch.randn(6, 64, 64)
    for arch in ("v1", "v2"):
        m = build_model(arch, 64, 4)
        logits, emb = m.features(x)
        assert logits.shape == (6, 4)
        logits.sum().backward()
        assert all(p.grad is not None for p in m.parameters() if p.requires_grad), arch
    assert emb.shape[0] == 6 and isinstance(build_model("v2", 64, 4), ActivityNetV2)
    assert isinstance(build_model("v1", 64, 4), ActivityNet)


def test_augment_keeps_shape_and_finite():
    x = torch.randn(16, 64, 64)
    for _ in range(20):
        y = augment(x)
        assert y.shape == x.shape and torch.isfinite(y).all()


def test_train_save_load_predict(tmp_path):
    root = str(tmp_path / "data")
    make_dataset(root, takes=3, seconds=8)
    out = str(tmp_path / "models")
    m = train(root, out, epochs=3, patience=3, log=lambda *_: None)
    t = m["test"]
    for key in ("accuracy", "macro_f1", "per_class", "confusion_matrix", "calibration", "selective"):
        assert key in t
    assert set(t["per_class"]) == set(LABELS)
    p = Predictor(os.path.join(out, "activity_model.pt"))        # weights_only load must accept the checkpoint
    assert p.arch == "v2" and p.temperature > 0
    assert p.preprocessing == "room_baseline_v1"
    ts = np.cumsum(np.full(600, 11.0))
    amp = np.random.default_rng(0).integers(40, 80, (600, 64)).astype(np.uint8)
    room_baseline = np.median(amp, axis=0).astype(np.float32)
    for _ in range(3):
        r = p.predict(ts, amp, room_baseline=room_baseline)
    assert r["label"] in LABELS + ("uncertain",) and abs(sum(r["probs"].values()) - 1) < 1e-6
    info = p.adapt_room(ts, amp, room_baseline=room_baseline)
    assert 0.33 <= p.gain <= 3.0 and "adapted" in info


def test_legacy_checkpoint_still_loads(tmp_path):
    ck = {"version": "old", "state_dict": ActivityNet(64, 4).state_dict(), "labels": list(LABELS),
          "feature_count": 64, "window_length": 64, "fs": 50, "scale": 1.0}
    path = str(tmp_path / "old.pt")
    torch.save(ck, path)
    p = Predictor(path)
    assert p.arch == "v1" and p.temperature == 1.0 and not p.novelty_ready
    ts = np.cumsum(np.full(600, 11.0))
    amp = np.random.default_rng(1).integers(40, 80, (600, 64)).astype(np.uint8)
    assert p.predict(ts, amp)["best"] in LABELS
