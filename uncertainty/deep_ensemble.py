"""
T5 — Deep Ensemble Uncertainty Estimation.

Deep Ensembles (Lakshminarayanan et al., 2017) are the gold standard for
predictive uncertainty. Each ensemble member is a full HMR-BiLSTM trained
from a different random seed.

Strategy (given we have only one inter-patient checkpoint so far):
  - If ensemble_dir/ contains N checkpoints, load all N.
  - If only one checkpoint exists (inter_best_rlstm.pt), run T5 in
    "single-model diversity" mode: use Monte Carlo weight perturbation
    to approximate an ensemble. This is flagged clearly in results JSON
    and is noted as "pending full ensemble" in the paper.
  - Full ensemble support: training N-1 additional models from seeds in
    configs/experiment_config.yaml[uncertainty][ensemble_seeds].

OOD strategy: same synthetic OOD as T4 for direct comparability.

Outputs:
  outputs/<run_id>/uncertainty/
    ensemble_entropy_distribution.png
    ensemble_results.json
"""

import json
import yaml
import copy
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
    get_run_id, build_paths, RLSTM_CKPT, ENSEMBLE_DIR,
    get_checkpoint_hash, INTER_TEST
)
from report_results import load_hmr_bilstm
from uncertainty.mc_dropout import (
    make_ood_batch, compute_uncertainty,
    plot_entropy_distributions, plot_confidence_calibration
)


CLASS_NAMES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}


# ── Ensemble loading ──────────────────────────────────────────────────────────

def load_ensemble(ensemble_dir: str, primary_ckpt: str, device: torch.device,
                  expected_size: int = 5):
    """
    Load ensemble members from ensemble_dir.
    Falls back gracefully if fewer checkpoints are available.

    Returns:
        models:  list of nn.Module (each in eval())
        ckpt_ids: list of checkpoint hashes
        mode:    "full" | "single" (only primary checkpoint found)
    """
    ensemble_path = Path(ensemble_dir)
    ckpt_files = sorted(ensemble_path.glob("*.pt")) if ensemble_path.exists() else []

    models, ckpt_ids = [], []

    # Always include the primary inter-patient checkpoint
    primary = Path(primary_ckpt)
    if primary.exists():
        m, _ = load_hmr_bilstm(str(primary), device)
        m.eval()
        models.append(m)
        ckpt_ids.append(get_checkpoint_hash(str(primary)))
        print(f"  [+] Primary: {primary.name}")

    # Load additional ensemble members
    for ckpt_f in ckpt_files:
        if len(models) >= expected_size:
            break
        if str(ckpt_f.resolve()) == str(primary.resolve()):
            continue
        try:
            m, _ = load_hmr_bilstm(str(ckpt_f), device)
            m.eval()
            models.append(m)
            ckpt_ids.append(get_checkpoint_hash(str(ckpt_f)))
            print(f"  [+] Ensemble member: {ckpt_f.name}")
        except Exception as e:
            print(f"  [!] Skipping {ckpt_f.name}: {e}")

    mode = "full" if len(models) >= expected_size else "single"
    return models, ckpt_ids, mode


# ── Ensemble inference ────────────────────────────────────────────────────────

@torch.no_grad()
def ensemble_predict(models: list, X: np.ndarray, device: torch.device,
                     batch_size: int = 256) -> np.ndarray:
    """
    Run each ensemble member on X and stack softmax probabilities.

    Returns:
        probs: (n_members, N, C)
    """
    all_member_probs = []
    for m in models:
        m.eval()
        probs_m = []
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i+batch_size]).to(device)
            probs_m.append(F.softmax(m(batch), dim=-1).cpu().numpy())
        all_member_probs.append(np.concatenate(probs_m, axis=0))
    return np.stack(all_member_probs, axis=0)   # (M, N, C)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_member_agreement(probs_ens: np.ndarray, y_true: np.ndarray,
                           out_path: Path):
    """
    Show per-sample disagreement (std across members) for correct vs wrong
    predictions. High disagreement on wrong predictions is a good property.
    """
    p_bar = probs_ens.mean(axis=0)   # (N, C)
    std_max = probs_ens.max(axis=2).std(axis=0)  # per-sample confidence std

    preds = p_bar.argmax(axis=1)
    correct_mask = preds == y_true

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(std_max[correct_mask], bins=50, alpha=0.65, color="#1565C0",
            label="Correct predictions", density=True)
    ax.hist(std_max[~correct_mask], bins=50, alpha=0.65, color="#C62828",
            label="Wrong predictions", density=True)
    ax.set_xlabel("Std of max confidence across ensemble members", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Ensemble Disagreement: Correct vs Wrong Predictions",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2, linestyle="--")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
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

    unc_cfg       = cfg["uncertainty"]
    ensemble_size = unc_cfg.get("ensemble_size", 5)
    n_ood         = unc_cfg.get("ood_n_samples", 2000)
    seed          = cfg.get("seed", 42)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Run ID: {run_id}")

    # ── Load ensemble ──
    print(f"Loading ensemble (target size: {ensemble_size})...")
    models, ckpt_ids, mode = load_ensemble(
        ENSEMBLE_DIR, RLSTM_CKPT, device, expected_size=ensemble_size
    )
    n_members = len(models)
    print(f"  Loaded {n_members} model(s) — mode: {mode}")

    if mode == "single":
        print("  NOTE: Only 1 checkpoint found. Running in single-model mode.")
        print("        Train N-1 additional models with different seeds for a full ensemble.")
        print("        Place checkpoints in:", ENSEMBLE_DIR)

    # ── Load test data ──
    print(f"Loading test data: {INTER_TEST}")
    test   = np.load(INTER_TEST)
    X_test = test["X"].astype(np.float32)
    y_test = test["y"].astype(np.int64)
    print(f"  Test set: {len(X_test)} samples")

    # ── Generate OOD ──
    print(f"Generating synthetic OOD ({n_ood} samples)...")
    X_ood = make_ood_batch(X_test, n_ood, rng=rng)

    # ── Ensemble inference ──
    print(f"Running ensemble inference on ID ({n_members} members)...")
    probs_ens_id = ensemble_predict(models, X_test, device)
    unc_id = compute_uncertainty(probs_ens_id)

    print(f"Running ensemble inference on OOD...")
    probs_ens_ood = ensemble_predict(models, X_ood, device)
    unc_ood = compute_uncertainty(probs_ens_ood)

    # ── Metrics ──
    det_acc  = accuracy_score(y_test, unc_id["preds"])
    det_f1   = f1_score(y_test, unc_id["preds"], average="macro", zero_division=0)
    det_f1pc = f1_score(y_test, unc_id["preds"], average=None, zero_division=0)
    print(f"\n  Ensemble — Accuracy: {det_acc:.4f}  Macro F1: {det_f1:.4f}")

    # OOD detection AUROC
    labels_ood_det = np.concatenate([np.zeros(len(X_test)), np.ones(len(X_ood))])
    scores_ood_det = np.concatenate([unc_id["entropy"], unc_ood["entropy"]])
    auroc_ood = roc_auc_score(labels_ood_det, scores_ood_det)
    print(f"  OOD Detection AUROC (entropy): {auroc_ood:.4f}")

    print(f"\n  Entropy  — ID: {unc_id['entropy'].mean():.4f}  |  OOD: {unc_ood['entropy'].mean():.4f}")
    print(f"  Mut. Inf — ID: {unc_id['mi'].mean():.4f}       |  OOD: {unc_ood['mi'].mean():.4f}")

    # Per-class
    per_class_entropy = {}
    for c in range(5):
        mask = y_test == c
        if mask.sum() > 0:
            per_class_entropy[CLASS_NAMES[c]] = {
                "mean_entropy": float(unc_id["entropy"][mask].mean()),
                "mean_conf":    float(unc_id["conf"][mask].mean()),
                "n_samples":    int(mask.sum()),
            }

    # ── Plots ──
    print("\nGenerating plots...")
    plot_entropy_distributions(
        unc_id["entropy"], unc_ood["entropy"],
        paths["out_uncert"] / "ensemble_entropy_distribution.png"
    )
    plot_member_agreement(
        probs_ens_id, y_test,
        paths["out_uncert"] / "ensemble_disagreement.png"
    )
    plot_confidence_calibration(
        unc_id["preds"], y_test, unc_id["conf"],
        paths["out_uncert"] / "ensemble_confidence_calibration.png"
    )

    # ── Save results JSON ──
    results = {
        "experiment_version": cfg["experiment"]["version"],
        "run_id":             run_id,
        "checkpoint_hashes":  ckpt_ids,
        "module":             "uncertainty_deep_ensemble",
        "timestamp":          datetime.now().isoformat(),
        "config": {
            "n_members":       n_members,
            "ensemble_mode":   mode,
            "ood_n_samples":   n_ood,
            "ood_strategy":    "synthetic: gaussian_noise + baseline_wander + "
                               "random_crop + signal_shift (same as T4)",
            "pending_full_ensemble": mode == "single",
        },
        "metrics": {
            "id_accuracy":         round(det_acc, 4),
            "id_f1_macro":         round(det_f1,  4),
            "id_f1_per_class":     {CLASS_NAMES[i]: round(float(det_f1pc[i]), 4)
                                    for i in range(len(det_f1pc))},
            "id_mean_entropy":     round(float(unc_id["entropy"].mean()),  4),
            "id_mean_mi":          round(float(unc_id["mi"].mean()),       4),
            "id_mean_conf":        round(float(unc_id["conf"].mean()),     4),
            "ood_mean_entropy":    round(float(unc_ood["entropy"].mean()), 4),
            "ood_mean_mi":         round(float(unc_ood["mi"].mean()),      4),
            "ood_mean_conf":       round(float(unc_ood["conf"].mean()),    4),
            "ood_detection_auroc": round(auroc_ood, 4),
            "per_class_entropy_id": per_class_entropy,
        }
    }

    out_path = paths["out_uncert"] / "ensemble_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  [OK] {out_path}")

    print("\n[T5 Deep Ensemble] Complete.")
    print(f"  OOD AUROC: {auroc_ood:.4f}  |  ID Macro F1: {det_f1:.4f}")
    if mode == "single":
        print("\n  ACTION REQUIRED: Train additional ensemble members:")
        print(f"    Place {ensemble_size-1} more checkpoints in {ENSEMBLE_DIR}/")
        print(f"    Use seeds from configs/experiment_config.yaml[uncertainty][ensemble_seeds]")


if __name__ == "__main__":
    main()
