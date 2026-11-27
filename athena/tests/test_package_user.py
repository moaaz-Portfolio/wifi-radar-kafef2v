"""User release smoke test: export, extract runtime and load a real torch model."""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import package_user  # noqa: E402
from athena.model import build_model  # noqa: E402


LABELS = ["empty_room", "person_static", "movement", "fall_detection"]


def test_exported_user_package_loads_model_after_extraction(tmp_path):
    app_dir = tmp_path / "athena"
    for relative in package_user.RUNTIME_FILES:
        source = package_user.APP / relative
        destination = app_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    model_dir = app_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "activity_model.pt"
    model_id = "package-smoke-test-model"
    checkpoint = {
        "version": "package-smoke-test",
        "model_id": model_id,
        "preprocessing": "room_baseline_v1",
        "arch": "v2",
        "state_dict": build_model("v2", 64, 4).state_dict(),
        "labels": LABELS,
        "feature_count": 64,
        "window_length": 64,
        "fs": 50,
        "scale": 1.0,
        "normalize": True,
        "confidence_threshold": 0.65,
        "temperature": 1.0,
        "novelty": None,
        "empty_noise_floor": 0.1,
    }
    torch.save(checkpoint, model_path)

    metrics = {
        "model_version": checkpoint["version"],
        "model_id": model_id,
        "model_sha256": package_user.sha256(model_path),
        "preprocessing": checkpoint["preprocessing"],
        "arch": checkpoint["arch"],
        "labels": LABELS,
        "test": {
            "per_class": {
                label: {"precision": 0.9, "recall": 0.9, "f1": 0.9}
                for label in LABELS
            }
        },
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    archive = tmp_path / "WiFi-Radar-User.zip"
    package_user.build("test", archive, researcher_approved=True, app_dir=app_dir)

    extracted = tmp_path / "extracted"
    with package_user.zipfile.ZipFile(archive) as zf:
        zf.extractall(extracted)

    user_root = extracted / "WiFi-Radar-User"
    assert (user_root / "athena" / "learning.py").is_file()
    assert (user_root / "models" / "activity_model.pt").is_file()

    code = (
        "import numpy as np; "
        "from athena.live import Predictor; "
        "p = Predictor('models/activity_model.pt'); "
        "assert p.labels == " + repr(LABELS) + "; "
        "ts = np.arange(120, dtype=np.float64) * 20.0; "
        "amp = np.tile(np.linspace(45, 85, 64, dtype=np.float32), (120, 1)); "
        "r = p.predict(ts, amp, room_baseline=np.zeros(64, dtype=np.float32)); "
        "assert r is not None and set(r['probs']) == set(p.labels); "
        "print('USER_PACKAGE_MODEL_LOAD_OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=user_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "USER_PACKAGE_MODEL_LOAD_OK" in result.stdout

    metrics["model_sha256"] = "not-the-model-hash"
    (model_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        package_user.validate_model(model_path, model_dir / "metrics.json")
