"""
T4 — MC Dropout Uncertainty Estimation (BN-safe).

Key design decisions:
  1. BN-safe activation: model.eval() first, then ONLY nn.Dropout layers are put
     back in train() mode. BatchNorm layers stay in eval mode → use frozen running
     stats → deterministic feature normalization across MC samples.
     Using model.train() globally would corrupt BN estimates with single-sample
     running stats and destroy reproducibility.

  2. OOD strategy: Synthetic OOD from the inter-patient test set itself via
     four transformations (Gaussian noise, random crop, baseline wander, signal shift).
     These mimic real-world distribution shifts without requiring PTB-XL download.

  3. Metrics:
     - Predictive entropy  H[p(y|x)] = -sum(p_bar * log(p_bar))
     - Mutual information  MI = H[p] - E_mc[H[p_t]]  (epistemic uncertainty)
     - AUROC (ID vs OOD) using entropy as anomaly score

Outputs:
  outputs/<run_id>/uncertainty/
    mc_entropy_distribution.png    ← ID vs OOD entropy histograms
    mc_confidence_calibration.png  ← mean MC confidence vs correctness
    mc_results.json                ← all metrics + checkpoint_hash
"""

import json
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from configs.paths import (
    get_run_id, build_paths, RLSTM_CKPT, get_checkpoint_hash, INTER_TEST
)
from report_results import load_hmr_bilstm


CLASS_NAMES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}


# ── BN-safe MC Dropout activation ────────────────────────────────────────────

def enable_mc_dropout(model: nn.Module) -> nn.Module:
    """
    Activate MC Dropout without corrupting BatchNorm statistics.

    model.eval() → all layers off (Dropout pass-through, BN uses running stats).
    Then ONLY nn.Dropout layers are switched to train() → Dropout is active.
    BatchNorm layers remain in eval() → frozen running stats.

    This is the ONLY correct way to run MC Dropout when model has BN.
    Using model.train() globally would update BN running stats per-sample
    and make attribution depend on batch order.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    return model


# ── OOD augmentation transforms ──────────────────────────────────────────────

def ood_gaussian_noise(X: np.ndarray, sigma: float = 0.5, rng=None) -> np.ndarray:
    """Add Gaussian noise with std σ (in normalized units)."""
    if rng is None:
        rng = np.random.default_rng(42)
    noise = rng.normal(0, sigma, X.shape).astype(np.float32)
    return X + noise


def ood_baseline_wander(X: np.ndarray, freq: float = 0.3, amp: float = 0.8,
                         rng=None) -> np.ndarray:
    """Sinusoidal baseline wander — common real-world ECG artifact."""
    T = X.shape[1]
    t = np.linspace(0, 2 * np.pi * freq * T / 187, T).astype(np.float32)
    wander = amp * np.sin(t)  # (T,)
    return X + wander[None, :, None]


def ood_random_crop_pad(X: np.ndarray, crop_frac: float = 0.15,
                         rng=None) -> np.ndarray:
    """Randomly zero-pad (crop) a fraction of the signal from start or end."""
    if rng is None:
        rng = np.random.default_rng(42)
    T = X.shape[1]
    n_crop = int(T * crop_frac)
    X_out = X.copy()
    for i in range(len(X)):
        start = bool(rng.integers(0, 2))
        if start:
            X_out[i, :n_crop, :] = 0.0
        else:
            X_out[i, T - n_crop:, :] = 0.0
    return X_out


def ood_signal_shift(X: np.ndarray, shift_frac: float = 0.2,
                      rng=None) -> np.ndarray:
    """Cyclically shift signal in time by up to shift_frac of length."""
    if rng is None:
        rng = np.random.default_rng(42)
    T = X.shape[1]
    max_shift = int(T * shift_frac)
    shifts = rng.integers(-max_shift, max_shift, size=len(X))
    X_out = np.stack([np.roll(X[i], s, axis=0) for i, s in enumerate(shifts)])
    return X_out


def make_ood_batch(X_id: np.ndarray, n_samples: int, rng=None) -> np.ndarray:
    """
    Build OOD set by applying all four transforms to a random subset of ID samples.
    Each transform is applied independently to n_samples//4 samples.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_each = n_samples // 4
    idx = rng.choice(len(X_id), n_each * 4, replace=True)
    X_base = X_id[idx].copy()

    parts = []
    parts.append(ood_gaussian_noise(X_base[:n_each], rng=rng))
    parts.append(ood_baseline_wander(X_base[n_each:2*n_each], rng=rng))
    parts.append(ood_random_crop_pad(X_base[2*n_each:3*n_each], rng=rng))
    parts.append(ood_signal_shift(X_base[3*n_each:], rng=rng))

    return np.concatenate(parts, axis=0)


# ── MC Inference ──────────────────────────────────────────────────────────────

@torch.no_grad()
def mc_predict(model: nn.Module, X: np.ndarray, n_samples: int,
               device: torch.device, batch_size: int = 128) -> np.ndarray:
    """
    Run n_samples stochastic forward passes with MC Dropout active.

    Returns:
        probs_mc: (n_samples, N, C) — softmax probabilities per MC sample
    """
    enable_mc_dropout(model)
    N = len(X)
    probs_all = []

    for _ in range(n_samples):
        probs_run = []
        for i in range(0, N, batch_size):
            batch = torch.from_numpy(X[i:i+batch_size]).to(device)
            logits = model(batch)
            probs_run.append(F.softmax(logits, dim=-1).cpu().numpy())
        probs_all.append(np.concatenate(probs_run, axis=0))

    return np.stack(probs_all, axis=0)   # (n_samples, N, C)


def compute_uncertainty(probs_mc: np.ndarray):
    """
    Compute predictive entropy and mutual information from MC samples.

    Args:
        probs_mc: (n_samples, N, C)

    Returns:
        dict with:
          - p_bar:   (N, C) — mean probability
          - entropy: (N,)   — predictive entropy H[p(y|x)]
          - mi:      (N,)   — mutual information (epistemic uncertainty)
          - conf:    (N,)   — max of p_bar (mean MC confidence)
          - preds:   (N,)   — argmax of p_bar
    """
    p_bar = probs_mc.mean(axis=0)                        # (N, C)
    eps = 1e-8

    # Predictive entropy: H[E[p(y|x,w)]]
    entropy = -(p_bar * np.log(p_bar + eps)).sum(axis=1)  # (N,)

    # Expected entropy: E[H[p(y|x,w)]] — aleatoric uncertainty
    expected_entropy = -(probs_mc * np.log(probs_mc + eps)).sum(axis=2).mean(axis=0)

    # Mutual information (epistemic) = total − aleatoric
    mi = entropy - expected_entropy
    mi = np.clip(mi, 0, None)

    conf = p_bar.max(axis=1)
    preds = p_bar.argmax(axis=1)

    return {
        "p_bar":   p_bar,
        "entropy": entropy,
        "mi":      mi,
        "conf":    conf,
        "preds":   preds,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_entropy_distributions(entropy_id: np.ndarray, entropy_ood: np.ndarray,
                                out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(entropy_id,  bins=60, alpha=0.65, color="#1565C0",
            label=f"In-Distribution (MIT-BIH inter-test, n={len(entropy_id)})",
            density=True)
    ax.hist(entropy_ood, bins=60, alpha=0.65, color="#C62828",
            label=f"OOD (Synthetic augmentation, n={len(entropy_ood)})",
            density=True)
    ax.set_xlabel("Predictive Entropy  H[p(y|x)]", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("MC Dropout: ID vs OOD Predictive Entropy", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out_path.name}")


def plot_confidence_calibration(preds: np.ndarray, labels: np.ndarray,
                                 conf: np.ndarray, out_path: Path,
                                 n_bins: int = 10):
    """
    Confidence-accuracy plot: mean confidence vs mean accuracy per bin.
    A well-calibrated model produces a diagonal line.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_acc, bin_conf, bin_cnt = [], [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() > 0:
            bin_acc.append((preds[mask] == labels[mask]).mean())
            bin_conf.append(conf[mask].mean())
            bin_cnt.append(mask.sum())

    bin_acc  = np.array(bin_acc)
    bin_conf = np.array(bin_conf)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.bar(bin_conf, bin_acc, width=0.08, alpha=0.7, color="#7B1FA2",
           label="MC Dropout accuracy per bin")
    ax.plot(bin_conf, bin_acc, "o-", color="#4A148C", linewidth=1.5)
    ax.set_xlabel("Mean MC Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("MC Dropout Confidence-Accuracy Calibration", fontsize=14,
                 fontweight="bold")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    config_path = Path("configs/experiment_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_id = get_run_id(cfg)
    paths  = build_paths(run_id)
    paths["out_uncert"].mkdir(parents=True, exist_ok=True)

    unc_cfg    = cfg["uncertainty"]
    n_samples  = unc_cfg.get("mc_samples", 50)
    n_ood      = unc_cfg.get("ood_n_samples", 2000)
    seed       = cfg.get("seed", 42)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Run ID: {run_id}")
    print(f"MC samples: {n_samples} | OOD samples: {n_ood}")

    # ── Load model ──
    print("Loading model...")
    model, _ = load_hmr_bilstm(RLSTM_CKPT, device)

    # ── Load test data ──
    print(f"Loading test data: {INTER_TEST}")
    test   = np.load(INTER_TEST)
    X_test = test["X"].astype(np.float32)
    y_test = test["y"].astype(np.int64)
    print(f"  Test set: {len(X_test)} samples")

    # ── Generate OOD data ──
    print(f"Generating synthetic OOD ({n_ood} samples)...")
    X_ood = make_ood_batch(X_test, n_ood, rng=rng)
    print(f"  OOD set: {len(X_ood)} samples")

    # ── MC Dropout on ID ──
    print(f"Running {n_samples} MC passes on ID test set...")
    probs_mc_id = mc_predict(model, X_test, n_samples, device)
    unc_id = compute_uncertainty(probs_mc_id)

    # ── MC Dropout on OOD ──
    print(f"Running {n_samples} MC passes on OOD set...")
    probs_mc_ood = mc_predict(model, X_ood, n_samples, device)
    unc_ood = compute_uncertainty(probs_mc_ood)

    # ── Standard (deterministic) evaluation ──
    model.eval()
    det_acc  = accuracy_score(y_test, unc_id["preds"])
    det_f1   = f1_score(y_test, unc_id["preds"], average="macro", zero_division=0)
    det_f1pc = f1_score(y_test, unc_id["preds"], average=None, zero_division=0)
    print(f"\n  Deterministic (mean MC) — Accuracy: {det_acc:.4f}  Macro F1: {det_f1:.4f}")

    # ── OOD detection: entropy as anomaly score ──
    labels_ood_det = np.concatenate([
        np.zeros(len(X_test)),
        np.ones(len(X_ood))
    ])
    scores_ood_det = np.concatenate([unc_id["entropy"], unc_ood["entropy"]])
    auroc_ood = roc_auc_score(labels_ood_det, scores_ood_det)
    print(f"  OOD Detection AUROC (entropy): {auroc_ood:.4f}")

    # ── Uncertainty statistics ──
    mean_entropy_id  = float(unc_id["entropy"].mean())
    mean_entropy_ood = float(unc_ood["entropy"].mean())
    mean_mi_id       = float(unc_id["mi"].mean())
    mean_mi_ood      = float(unc_ood["mi"].mean())
    mean_conf_id     = float(unc_id["conf"].mean())
    mean_conf_ood    = float(unc_ood["conf"].mean())

    print(f"\n  Entropy  — ID: {mean_entropy_id:.4f}  |  OOD: {mean_entropy_ood:.4f}")
    print(f"  Mut. Inf — ID: {mean_mi_id:.4f}       |  OOD: {mean_mi_ood:.4f}")
    print(f"  Conf.    — ID: {mean_conf_id:.4f}      |  OOD: {mean_conf_ood:.4f}")

    # ── Per-class entropy analysis ──
    per_class_entropy = {}
    for c in range(5):
        mask = y_test == c
        if mask.sum() > 0:
            per_class_entropy[CLASS_NAMES[c]] = {
                "mean_entropy": float(unc_id["entropy"][mask].mean()),
                "mean_conf":    float(unc_id["conf"][mask].mean()),
                "n_samples":    int(mask.sum()),
            }

    print("\n  Per-class entropy (ID):")
    for cn, stats in per_class_entropy.items():
        print(f"    {cn}: entropy={stats['mean_entropy']:.4f}  "
              f"conf={stats['mean_conf']:.4f}  n={stats['n_samples']}")

    # ── Plots ──
    print("\nGenerating plots...")
    plot_entropy_distributions(
        unc_id["entropy"], unc_ood["entropy"],
        paths["out_uncert"] / "mc_entropy_distribution.png"
    )
    plot_confidence_calibration(
        unc_id["preds"], y_test, unc_id["conf"],
        paths["out_uncert"] / "mc_confidence_calibration.png"
    )

    # ── Save results JSON ──
    results = {
        "experiment_version": cfg["experiment"]["version"],
        "run_id":             run_id,
        "checkpoint_hash":    get_checkpoint_hash(RLSTM_CKPT),
        "module":             "uncertainty_mc_dropout",
        "timestamp":          datetime.now().isoformat(),
        "config": {
            "mc_samples":   n_samples,
            "ood_n_samples": n_ood,
            "ood_strategy": "synthetic: gaussian_noise + baseline_wander + "
                            "random_crop + signal_shift",
            "bn_safe_note": (
                "model.eval() first, then only nn.Dropout layers set to train(). "
                "BatchNorm layers remain in eval() with frozen running statistics."
            ),
        },
        "metrics": {
            "id_accuracy":          round(det_acc,  4),
            "id_f1_macro":          round(det_f1,   4),
            "id_f1_per_class":      {CLASS_NAMES[i]: round(float(det_f1pc[i]), 4)
                                     for i in range(len(det_f1pc))},
            "id_mean_entropy":      round(mean_entropy_id,  4),
            "id_mean_mi":           round(mean_mi_id,       4),
            "id_mean_conf":         round(mean_conf_id,     4),
            "ood_mean_entropy":     round(mean_entropy_ood, 4),
            "ood_mean_mi":          round(mean_mi_ood,      4),
            "ood_mean_conf":        round(mean_conf_ood,    4),
            "ood_detection_auroc":  round(auroc_ood, 4),
            "per_class_entropy_id": per_class_entropy,
        }
    }

    out_path = paths["out_uncert"] / "mc_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  [OK] {out_path}")

    print("\n[T4 MC Dropout] Complete.")
    print(f"  OOD AUROC: {auroc_ood:.4f}  |  ID Macro F1: {det_f1:.4f}")


if __name__ == "__main__":
    main()
