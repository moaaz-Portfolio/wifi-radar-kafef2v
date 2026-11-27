"""Run a trained model on the live stream - and say "I don't know" when it should."""
from collections import deque

import numpy as np

from . import learning as L
from .data import make_windows, normalize_window, resample


class Predictor:
    def __init__(self, path):
        import torch
        from .model import build_model
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.torch = torch
        self.labels = ck["labels"]
        self.win = int(ck["window_length"])
        self.fs = float(ck["fs"])
        self.scale = float(ck["scale"])
        self.threshold = float(ck.get("confidence_threshold", 0.65))
        self.version = ck.get("version", "?")
        self.arch = ck.get("arch", "v1")                   # checkpoints from before v2 have no 'arch'
        self.normalize = bool(ck.get("normalize", True))
        self.preprocessing = ck.get(
            "preprocessing",
            "legacy_window_center_v0" if self.normalize else "raw_amplitude_v0",
        )
        if self.preprocessing not in ("room_baseline_v1", "legacy_window_center_v0",
                                      "raw_amplitude_v0", "raw_amplitude_v1"):
            raise ValueError(f"unsupported model preprocessing: {self.preprocessing}")
        self.temperature = float(ck.get("temperature", 1.0))
        self.novelty = L.NoveltyDetector.from_dict(ck.get("novelty")) if self.arch != "v1" else L.NoveltyDetector()
        self.adapter = L.RoomAdapter(float(ck.get("empty_noise_floor", 0.0)))
        self.gain = 1.0
        self.model = build_model(self.arch, int(ck["feature_count"]), len(self.labels))
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self._decider = L.HysteresisDecider(k=4, switch_frames=2)
        self._novel_hist = deque(maxlen=3)

    @property
    def seconds_needed(self):
        return self.win / self.fs + 1.0

    @property
    def novelty_ready(self):
        return self.novelty.ready

    def set_gain(self, gain):
        self.gain = float(gain)
        self.reset()

    def reset(self):
        self._decider.reset()
        self._novel_hist.clear()

    def _preprocess_windows(self, windows, room_baseline=None):
        if self.preprocessing == "room_baseline_v1":
            if room_baseline is None:
                return None
            return np.stack([normalize_window(w, room_baseline) for w in windows])
        if self.preprocessing == "legacy_window_center_v0":
            return np.stack([w - w.mean(axis=0, keepdims=True) for w in windows])
        return windows

    def adapt_room(self, ts, amp, room_baseline=None):
        """Adapt the model using empty-room samples referenced to that room's saved baseline."""
        series = resample(ts, amp, self.fs)
        wins = make_windows(series, self.win, max(self.win // 4, 1))
        if len(wins) < 5 or self.adapter.train_floor <= 0:
            return {"adapted": False, "gain": self.gain,
                    "reason": "not enough empty-room data or the model stores no noise floor"}
        wins = self._preprocess_windows(wins, room_baseline)
        if wins is None:
            return {"adapted": False, "gain": self.gain,
                    "reason": "empty-room calibration is required for this model"}
        floor = L.window_noise_floor(wins)
        self.set_gain(self.adapter.gain(floor))
        return {"adapted": True, "gain": self.gain, "room_floor": floor,
                "train_floor": self.adapter.train_floor}

    def predict(self, ts, amp, room_baseline=None):
        series = resample(ts, amp, self.fs)
        if len(series) < self.win:
            return None
        w = series[-self.win:]
        processed = self._preprocess_windows(w[None, :, :], room_baseline)
        if processed is None:
            return None
        x = processed[0] * self.gain / self.scale
        with self.torch.no_grad():
            logits, emb = self.model.features(self.torch.from_numpy(x[None]).float())
        probs = L.softmax(logits[0].numpy().astype(np.float64) / self.temperature)
        current, avg = self._decider.update(probs)
        conf = float(avg[current])

        novel_now, novelty_score = False, None
        if self.novelty.ready:
            novelty_score = float(self.novelty.score(emb[0].numpy())[0])
            novel_now = novelty_score > self.novelty.threshold
        self._novel_hist.append(novel_now)
        novel = sum(self._novel_hist) >= 2                    # two of the last three: ignore one-off glitches

        reason = "ok"
        if novel:
            reason = "novel_input"                            # looks like nothing seen in training
        elif conf < self.threshold:
            reason = "low_confidence"
        label = self.labels[current] if reason == "ok" else "uncertain"
        return {"label": label,
                "best": self.labels[current], "confidence": conf, "reason": reason,
                "novelty": novelty_score,
                "probs": {l: float(v) for l, v in zip(self.labels, avg)}}


def motion_energy(amp):
    """Cheap, model-free 'how much is the signal moving' number for the last second."""
    if len(amp) < 5:
        return 0.0
    return float(np.mean(np.std(amp, axis=0)))
