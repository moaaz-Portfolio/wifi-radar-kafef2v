"""Train on your own recordings (v2: deep model, EMA weights, calibration, novelty detection)."""
import copy
import hashlib
import io
import math
import os
import random
import time
import uuid

import numpy as np

from . import learning as L
from .data import FEATURES, load_dataset, write_json

CONF_THRESHOLD = 0.65


def confusion(y, p, n):
    m = np.zeros((n, n), dtype=np.int64)
    for a, b in zip(y, p):
        m[a, b] += 1
    return m


def scores(m):
    tp = np.diag(m).astype(float)
    prec = tp / np.maximum(m.sum(0), 1)
    rec = tp / np.maximum(m.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return prec, rec, f1, float(tp.sum() / max(m.sum(), 1))


def _xy(part):
    """A split is (X, y) or (X, y, recording_ids)."""
    return part[0], part[1], (part[2] if len(part) > 2 else None)


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def augment(x, strength=1.0):
    """Physical-style augmentation on a (B, T, F) batch already scaled to ~unit std.
    Every transform models something that really differs between rooms / days / hardware:
      gain           - distance to router, antenna gain, AGC (log-uniform 0.5x..2x)
      subcarrier tilt- frequency-selective fading differs per room
      time warp      - people move faster or slower
      subcarrier drop- a few bad subcarriers
      time mask      - short packet-loss bursts
      noise          - receiver noise
    """
    import torch
    import torch.nn.functional as F
    B, T, C = x.shape
    dev = x.device
    s = strength
    gain = torch.exp((torch.rand(B, 1, 1, device=dev) * 2 - 1) * math.log(2.0) * s)
    tilt = F.avg_pool1d(torch.randn(B, 1, C, device=dev), 5, 1, 2, count_include_pad=False) * 0.3 * s
    x = x * gain * (1.0 + tilt)                                          # (B,1,C) broadcasts over time
    # time warp (one speed factor per batch keeps it cheap), then crop / pad back to T
    speed = 1.0 + (random.random() * 2 - 1) * 0.15 * s
    new_t = max(8, int(round(T * speed)))
    if new_t != T:
        y = F.interpolate(x.transpose(1, 2), size=new_t, mode="linear", align_corners=False)
        if new_t > T:
            o = random.randint(0, new_t - T)
            y = y[:, :, o:o + T]
        else:
            y = F.pad(y, (0, T - new_t), mode="replicate")
        x = y.transpose(1, 2)
    drop_p = 0.12 * s * (torch.rand(B, 1, 1, device=dev) < 0.5).float()
    x = x * (torch.rand(B, 1, C, device=dev) >= drop_p).float()
    start = torch.randint(0, T, (B, 1), device=dev)
    length = torch.randint(1, 9, (B, 1), device=dev)
    t_idx = torch.arange(T, device=dev).unsqueeze(0)
    cut = ((t_idx >= start) & (t_idx < start + length) & (torch.rand(B, 1, device=dev) < 0.3 * s)).unsqueeze(2)
    x = x.masked_fill(cut, 0.0)
    return x + torch.randn_like(x) * (0.02 + 0.06 * torch.rand(B, 1, 1, device=dev)) * s


def _param_groups(model, wd):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim < 2 or name.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def _lr_at(step, total, warmup, floor=0.01):
    if step < warmup:
        return (step + 1) / warmup
    prog = (step - warmup) / max(total - warmup, 1)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


def train(root, out_dir, epochs=30, fs=50, win=64, hop=16,
          log=print, progress=None, should_stop=lambda: False,
          dataset=None, normalize=True, save=True, model_filename="activity_model.pt",
          model_version="athena-own-2", metrics_filename="metrics.json",
          arch="v2", seed=7, patience=12, lr=1.5e-3, weight_decay=0.05, ema_decay=0.995,
          mixup_prob=0.25, batch_size=128, target_accuracy=0.95):
    """dataset=(labels, {train,val,test: (X, y[, recording_ids])}, counts) lets crossval.py supply its own
    split; save=False evaluates without writing any file. Returns the metrics dict."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from .model import build_model

    seed_everything(seed)
    labels, data, counts = dataset if dataset is not None else load_dataset(root, fs, win, hop, normalize, return_groups=True)
    (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, gte) = _xy(data["train"]), _xy(data["val"]), _xy(data["test"])
    n_cls = len(labels)
    log(f"recordings train/val/test: {counts['train']}/{counts['val']}/{counts['test']}  "
        f"windows: {len(Xtr)}/{len(Xva)}/{len(Xte)}  classes: {labels}  arch: {arch}")

    scale = float(Xtr.std()) + 1e-6          # one global number, keeps relative motion strength
    empty_idx = labels.index("empty_room") if "empty_room" in labels else None
    empty_floor = L.window_noise_floor(Xtr[ytr == empty_idx]) if empty_idx is not None else 0.0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32) / scale)
    Xtr_t, Xva_t, Xte_t = T(Xtr).to(dev), T(Xva).to(dev), T(Xte).to(dev)
    ytr_t = torch.from_numpy(ytr).to(dev)

    model = build_model(arch, FEATURES, n_cls).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    cnt = np.bincount(ytr, minlength=n_cls).astype(float)
    w = torch.tensor(cnt.sum() / np.maximum(cnt, 1) / n_cls, dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(_param_groups(model, weight_decay), lr=lr, betas=(0.9, 0.98))
    steps_per_epoch = max(1, math.ceil(len(Xtr_t) / batch_size))
    total_steps, warmup = epochs * steps_per_epoch, min(3, max(epochs // 10, 1)) * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr_at(s, total_steps, warmup))

    def infer(net, X):
        net.eval()
        lg, em = [], []
        with torch.no_grad():
            for i in range(0, len(X), 512):
                a, b = net.features(X[i:i + 512])
                lg.append(a.float().cpu().numpy())
                em.append(b.float().cpu().numpy())
        return np.concatenate(lg), np.concatenate(em)

    best_state, best_key, bad, step = None, (-1.0, 0.0), 0, 0
    for ep in range(1, epochs + 1):
        if should_stop():
            log("stopped by user")
            break
        model.train()
        perm = torch.randperm(len(Xtr_t), device=dev)
        tot = 0.0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:                          # BatchNorm cannot train on a single sample
                continue
            x, y = augment(Xtr_t[idx]), ytr_t[idx]
            opt.zero_grad(set_to_none=True)
            if random.random() < mixup_prob:
                lam = float(np.random.beta(0.4, 0.4))
                j = torch.randperm(len(x), device=dev)
                out = model(lam * x + (1 - lam) * x[j])
                loss = lam * lossf(out, y) + (1 - lam) * lossf(out, y[j])
            else:
                loss = lossf(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            step += 1
            d = min(ema_decay, (1 + step) / (10 + step))          # EMA warm-up
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), model.parameters()):
                    pe.mul_(d).add_(p.detach(), alpha=1 - d)
                for be, b in zip(ema.buffers(), model.buffers()):
                    be.copy_(b)
            tot += loss.item() * len(idx)
        lg, _ = infer(ema, Xva_t)
        pred = lg.argmax(1)
        m = confusion(yva, pred, n_cls)
        _, _, f1, acc = scores(m)
        mf1, vnll = float(f1.mean()), L.nll(lg, yva)
        key = (round(mf1, 6), -vnll)
        if key > best_key:
            best_key, bad = key, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in ema.state_dict().items()})
        else:
            bad += 1
        log(f"epoch {ep:>2}/{epochs}  loss {tot / len(perm):.3f}  val acc {acc:.3f}  "
            f"macro-F1 {mf1:.3f}  val NLL {vnll:.3f}  lr {opt.param_groups[0]['lr']:.2e}")
        if progress:
            progress(ep, epochs)
        if bad >= patience:
            log(f"early stop: no validation improvement for {patience} epochs")
            break

    if best_state is None:
        raise RuntimeError("training did not run")
    ema.load_state_dict(best_state)

    # ---- calibration, abstain threshold and novelty detector (all fitted on VALIDATION data only)
    val_logits, val_emb = infer(ema, Xva_t)
    temperature = L.fit_temperature(val_logits, yva)
    val_probs = L.softmax(val_logits / temperature)
    threshold, thr_info = L.select_threshold(val_probs, yva, target_accuracy, default=CONF_THRESHOLD)
    novelty = None
    if arch != "v1":
        _, tr_emb = infer(ema, Xtr_t)
        novelty = L.NoveltyDetector().fit(tr_emb, ytr, n_cls)
        novelty.calibrate(val_emb)

    # ---- independent test report
    te_logits, te_emb = infer(ema, Xte_t)
    te_probs = L.softmax(te_logits / temperature)
    te_pred = te_logits.argmax(1)
    mt = confusion(yte, te_pred, n_cls)
    pr, rc, f1, acc = scores(mt)
    answered = te_probs.max(1) >= threshold
    novel = novelty.is_novel(te_emb) if novelty is not None else np.zeros(len(yte), dtype=bool)
    answered &= ~novel
    present = [k for k in range(n_cls) if mt[k].sum() > 0]
    calibration = {
        "temperature": temperature,
        "nll_before": L.nll(te_logits, yte), "nll_after": L.nll(te_logits, yte, temperature),
        "ece_before": L.expected_calibration_error(L.softmax(te_logits), yte),
        "ece_after": L.expected_calibration_error(te_probs, yte),
        "brier": L.brier(te_probs, yte),
    }
    selective = {
        "confidence_threshold": threshold, "threshold_selection": thr_info,
        "coverage": float(answered.mean()),
        "accuracy_when_answering": float((te_pred[answered] == yte[answered]).mean()) if answered.any() else None,
        "novel_fraction": float(novel.mean()),
        "novelty_threshold": novelty.threshold if novelty is not None else None,
    }
    ci = L.block_bootstrap_ci(yte, te_pred, gte, n_cls) if gte is not None else None

    model_id = uuid.uuid4().hex
    preprocessing = "room_baseline_v1" if normalize else "raw_amplitude_v1"
    ckpt = {
        "version": model_version, "model_id": model_id,
        "preprocessing": preprocessing, "arch": arch,
        "state_dict": {k: v.cpu() for k, v in ema.state_dict().items()},
        "labels": labels, "feature_count": FEATURES, "window_length": win,
        "fs": fs, "scale": scale, "confidence_threshold": float(threshold), "normalize": normalize,
        "temperature": float(temperature),
        "novelty": novelty.to_dict() if novelty is not None else None,
        "empty_noise_floor": float(empty_floor), "seed": seed,
    }

    checkpoint_buffer = io.BytesIO()
    torch.save(ckpt, checkpoint_buffer)
    model_bytes = checkpoint_buffer.getvalue()
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()

    metrics = {
        "model_version": ckpt["version"], "model_id": model_id,
        "model_sha256": model_sha256, "preprocessing": preprocessing,
        "arch": arch, "normalize": normalize, "labels": labels, "fs": fs,
        "window_length": win, "seed": seed,
        "recordings": counts, "windows": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)},
        "best_val_macro_f1": best_key[0],
        "test": {"accuracy": acc, "macro_f1": float(f1[present].mean()),
                 "balanced_accuracy": float(rc[present].mean()),
                 "per_class": {l: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i])}
                               for i, l in enumerate(labels)},
                 "confusion_matrix": mt.tolist(),
                 "confidence_interval_95": ci,
                 "calibration": calibration, "selective": selective},
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "warning": "Test recordings come from the same room/person as training unless you "
                   "recorded in several rooms. Re-test in a new room before trusting it.",
    }
    # Same keys as before so package_user.py / app.py keep working, but macro_f1 above now ignores
    # classes that have no test windows instead of counting them as 0.
    msg = (f"TEST accuracy {acc:.3f}  macro-F1 {metrics['test']['macro_f1']:.3f}  "
           f"ECE {calibration['ece_before']:.3f}->{calibration['ece_after']:.3f} (T={temperature:.2f})  "
           f"answers {selective['coverage']:.0%} of windows")
    if save:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, model_filename)
        temporary_path = path + ".tmp"
        with open(temporary_path, "wb") as stream:
            stream.write(model_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        write_json(os.path.join(out_dir, metrics_filename), metrics)
        log(f"{msg}  -> saved {path} (model id {model_id})")
    else:
        log(msg)
    return metrics
