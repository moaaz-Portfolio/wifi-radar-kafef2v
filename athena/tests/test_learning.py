"""Honest-learning toolkit checks. numpy only.   python -m pytest tests/test_learning.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from athena import learning as L  # noqa: E402


def _overconfident(n=600, k=4, seed=0, noise=2.0, scale=6.0):
    """Logits that are right ~70% of the time but shouted with huge margins."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    logits = rng.normal(0, noise, (n, k))
    logits[np.arange(n), y] += 2.0
    return logits * scale, y


def test_temperature_fixes_overconfidence():
    logits, y = _overconfident()
    T = L.fit_temperature(logits, y)
    assert T > 2.0, T                                    # must cool the scores down
    before = L.expected_calibration_error(L.softmax(logits), y)
    after = L.expected_calibration_error(L.softmax(logits / T), y)
    assert after < before * 0.5 and after < 0.08, (before, after)
    assert L.nll(logits, y, T) < L.nll(logits, y, 1.0)


def test_temperature_refuses_tiny_data():
    logits, y = _overconfident(n=10)
    assert L.fit_temperature(logits, y) == 1.0


def test_calibrated_model_keeps_temperature_near_one():
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 2.0, (3000, 4))             # labels are DRAWN from these probabilities,
    p = L.softmax(logits)                              # so the model is perfectly calibrated by design
    y = np.array([rng.choice(4, p=row) for row in p])
    T = L.fit_temperature(logits, y)
    assert 0.8 < T < 1.25, T


def test_select_threshold_reaches_target_or_says_not():
    rng = np.random.default_rng(2)
    n = 2000
    y = rng.integers(0, 4, n)
    correct = rng.random(n) < 0.8
    conf = np.where(correct, rng.uniform(0.7, 1.0, n), rng.uniform(0.25, 0.6, n))
    probs = np.full((n, 4), 0.0)
    pred = np.where(correct, y, (y + 1) % 4)
    for i in range(n):
        probs[i] = (1 - conf[i]) / 3
        probs[i, pred[i]] = conf[i]
    thr, info = L.select_threshold(probs, y, target_accuracy=0.95)
    assert info["reached_target"] and 0.5 <= thr <= 0.8, (thr, info)
    assert info["accuracy"] >= 0.95 and info["coverage"] >= 0.3
    coin = np.full((200, 4), 0.25)
    thr2, info2 = L.select_threshold(coin, rng.integers(0, 4, 200), target_accuracy=0.95)
    assert not info2["reached_target"] and thr2 == 0.65


def _blobs(n=400, d=16, seed=3):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 4, (4, d))
    y = rng.integers(0, 4, n)
    return centres[y] + rng.normal(0, 1, (n, d)), y, centres


def test_novelty_flags_unfamiliar_but_not_familiar():
    e, y, centres = _blobs()
    det = L.NoveltyDetector().fit(e, y, 4)
    ev, _, _ = _blobs(seed=4)
    ev = centres[np.random.default_rng(4).integers(0, 4, 400)] + np.random.default_rng(9).normal(0, 1, (400, 16))
    det.calibrate(ev)
    assert det.ready
    familiar = centres[np.random.default_rng(5).integers(0, 4, 300)] + np.random.default_rng(6).normal(0, 1, (300, 16))
    assert det.is_novel(familiar).mean() < 0.05
    alien = np.random.default_rng(7).normal(0, 1, (300, 16)) * 3 + 25      # a "room" the net never saw
    assert det.is_novel(alien).mean() > 0.95


def test_novelty_roundtrip_and_disabled_when_data_poor():
    e, y, _ = _blobs()
    det = L.NoveltyDetector().fit(e, y, 4)
    det.calibrate(e)
    det2 = L.NoveltyDetector.from_dict(det.to_dict())
    assert np.allclose(det.score(e[:20]), det2.score(e[:20]))
    poor = L.NoveltyDetector().fit(e[:5], y[:5], 4)
    assert poor.calibrate(e[:5]) is None and not poor.ready
    assert not poor.is_novel(e[:3]).any()           # disabled detector never claims novelty
    assert L.NoveltyDetector.from_dict(None).to_dict() is None


def test_novelty_skips_absent_class():
    e, y, _ = _blobs()
    y = np.where(y == 3, 2, y)                      # class 3 never appears
    det = L.NoveltyDetector().fit(e, y, 4)
    assert np.isnan(det.means[3]).all()
    assert np.isfinite(det.score(e)).all()
    assert L.NoveltyDetector.from_dict(det.to_dict()).score(e[:5]).shape == (5,)


def test_block_bootstrap_resamples_recordings():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(12), 50)
    y = rng.integers(0, 4, len(groups))
    pred = np.where(rng.random(len(y)) < 0.8, y, (y + 1) % 4)
    ci = L.block_bootstrap_ci(y, pred, groups, 4, n_boot=300)
    assert ci["recordings"] == 12
    lo, hi = ci["accuracy"]
    assert lo < 0.8 < hi and hi - lo < 0.2
    assert L.block_bootstrap_ci(y[:100], pred[:100], groups[:100], 4) is None     # 2 recordings: no fake CI


def test_room_adapter_aligns_noise_floor_and_clamps():
    rng = np.random.default_rng(0)
    quiet = rng.normal(0, 1.0, (50, 64, 64)).astype(np.float32)
    loud = quiet * 2.0
    f_quiet, f_loud = L.window_noise_floor(quiet), L.window_noise_floor(loud)
    assert abs(f_loud / f_quiet - 2.0) < 1e-3
    ad = L.RoomAdapter(f_quiet)
    assert abs(ad.gain(f_loud) - 0.5) < 1e-3
    assert ad.gain(f_quiet * 100) == ad.lo and ad.gain(f_quiet / 100) == ad.hi
    assert ad.gain(0.0) == 1.0 and L.window_noise_floor(np.zeros((0, 64, 64))) == 0.0


def test_hysteresis_blocks_flicker_but_follows_real_change():
    d = L.HysteresisDecider(k=3, switch_frames=2)
    a, b = np.array([.9, .05, .03, .02]), np.array([.05, .9, .03, .02])
    for _ in range(4):
        cur, _ = d.update(a)
    assert cur == 0
    cur, _ = d.update(b)                       # one odd frame
    assert cur == 0
    for _ in range(6):
        cur, avg = d.update(b)
    assert cur == 1 and abs(avg.sum() - 1) < 1e-9


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
