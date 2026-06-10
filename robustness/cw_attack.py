"""
T6 — Carlini & Wagner (C&W) L2 Attack.

C&W is an optimization-based attack that finds the minimum-norm adversarial
perturbation. Unlike FGSM/PGD which move in a fixed direction, C&W solves:

    minimize  ||δ||_2 + c · f(x + δ)
    s.t.      x + δ ∈ [data_min, data_max]

where f is a custom loss that drives the model to predict any wrong class.

Three-stage binary search for c (the trade-off parameter):
  - Stage 1: find c that makes the attack succeed at all
  - Stage 2: refine c to minimize perturbation norm
  - Stage 3: confirm best c found

Why C&W here:
  C&W is more powerful than PGD and bypasses many gradient masking defenses.
  If the model has low PGD-ASR but high C&W-ASR, it confirms gradient masking.
  This directly answers the P3 concern from the pre-validation report.

Outputs:
  outputs/<run_id>/robustness/
    cw_attack_results.json
    cw_perturbation_norms.png
    cw_asr_by_class.png
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
from sklearn.metrics import f1_score

from configs.paths import (
    get_run_id, build_paths, RLSTM_CKPT, get_checkpoint_hash, INTER_TEST
)
from report_results import load_hmr_bilstm


CLASS_NAMES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}


# ── C&W objective ─────────────────────────────────────────────────────────────

def cw_loss(logits: torch.Tensor, y_true: torch.Tensor,
            kappa: float = 0.0) -> torch.Tensor:
    """
    C&W f-function: encourages misclassification.

    f(x) = max(Z_t - max_{i≠t} Z_i + κ, 0)
    where Z_t = logit for true class.

    Attack succeeds when f(x) ≤ 0.
    κ (confidence): higher κ → more confident misclassification needed.
    """
    one_hot = F.one_hot(y_true, num_classes=logits.size(1)).float()
    correct_logit = (logits * one_hot).sum(dim=1)
    wrong_logit   = (logits * (1 - one_hot) - 1e9 * one_hot).max(dim=1).values
    return torch.clamp(correct_logit - wrong_logit + kappa, min=0)


# ── Single C&W attack pass ────────────────────────────────────────────────────

def cw_attack_batch(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                     c: float, lr: float = 0.01, max_steps: int = 500,
                     kappa: float = 0.0,
                     data_min: float = -5.0, data_max: float = 5.0,
                     device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    C&W L2 attack via change of variable: x_adv = tanh(w) (keeps in bounds).

    Args:
        c:         trade-off constant (larger → more focus on misclassification)
        lr:        Adam learning rate for perturbation optimization
        max_steps: optimization steps
        kappa:     confidence margin

    Returns:
        x_adv: (B, T, 1) adversarial examples, clamped to [data_min, data_max]
    """
    model.eval()
    x = x.to(device)
    y = y.to(device)
    B = x.size(0)

    # Change of variable: x_adv = (tanh(w) + 1) / 2 * (max - min) + min
    # So w = atanh(2*(x - min)/(max - min) - 1)
    # Clamp input to avoid atanh singularity at ±1
    scale = data_max - data_min
    x_norm = (x - data_min) / scale
    x_norm = x_norm.clamp(1e-6, 1 - 1e-6)
    w = torch.atanh(2 * x_norm - 1).detach().requires_grad_(True)

    optimizer = torch.optim.Adam([w], lr=lr)
    best_adv  = x.clone()
    best_l2   = torch.full((B,), float("inf"), device=device)

    for _ in range(max_steps):
        # Reconstruct x_adv from w
        x_adv = ((torch.tanh(w) + 1) / 2) * scale + data_min

        # L2 perturbation norm (per sample)
        l2 = ((x_adv - x) ** 2).sum(dim=(1, 2)).sqrt()

        # C&W objective
        logits = model(x_adv)
        attack_loss = cw_loss(logits, y, kappa)
        loss = l2.mean() + c * attack_loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update best adversarial examples where attack succeeded
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            success = preds != y
            improved = success & (l2 < best_l2)
            best_adv[improved] = x_adv[improved]
            best_l2[improved]  = l2[improved]

    return best_adv.detach()


# ── Stratified subset selection ───────────────────────────────────────────────

def select_stratified_subset(X: np.ndarray, y: np.ndarray,
                              n_total: int, rng) -> tuple:
    """Stratified subsample prioritising minority classes S, V, F."""
    priority = [1, 2, 3, 0, 4]
    classes  = np.unique(y)
    quota    = max(2, n_total // len(classes))

    indices = []
    for c in priority:
        if c not in classes:
            continue
        c_idx = np.where(y == c)[0]
        n_sel = min(quota, len(c_idx), n_total - len(indices))
        if n_sel > 0:
            indices.extend(rng.choice(c_idx, n_sel, replace=False))
        if len(indices) >= n_total:
            break

    return X[indices], y[indices], np.array(indices)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_perturbation_norms(l2_norms: dict, out_path: Path):
    """Box plot of L2 perturbation norms per class."""
    classes = [k for k in l2_norms if l2_norms[k]]
    data    = [l2_norms[k] for k in classes]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, labels=classes, patch_artist=True)
    colors = ["#1565C0", "#C62828", "#2E7D32", "#F57F17", "#6A1B9A"]
    for patch, color in zip(bp["boxes"], colors[:len(classes)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel("AAMI Class", fontsize=12)
    ax.set_ylabel("L2 Perturbation Norm", fontsize=12)
    ax.set_title("C&W L2 Attack: Perturbation Norms per Class", fontsize=13,
                 fontweight="bold")
    ax.grid(alpha=0.25, linestyle="--", axis="y")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out_path.name}")


def plot_asr_by_class(asr_per_class: dict, out_path: Path):
    """Bar chart: Attack Success Rate per AAMI class."""
    classes = list(asr_per_class.keys())
    rates   = [asr_per_class[c] for c in classes]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#B71C1C" if r > 0.5 else "#1565C0" for r in rates]
    bars = ax.bar(classes, [r * 100 for r in rates], color=colors, alpha=0.8, width=0.5)
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{r*100:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("AAMI Class", fontsize=12)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
    ax.set_title("C&W Attack Success Rate per Class", fontsize=13, fontweight="bold")
    ax.set_ylim([0, 110])
    ax.grid(alpha=0.25, linestyle="--", axis="y")
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
    paths["out_robust"].mkdir(parents=True, exist_ok=True)

    rob_cfg    = cfg["robustness"]
    cw_c       = rob_cfg.get("cw_c", 1e-4)
    cw_steps   = rob_cfg.get("cw_steps", 1000)
    seed       = cfg.get("seed", 42)

    # Limit subset size for CPU tractability
    n_eval = 200   # samples to attack (stratified)
    cw_lr  = 0.01
    kappa  = 0.0

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Run ID: {run_id}")
    print(f"C&W params: c={cw_c}, steps={cw_steps}, n_eval={n_eval}")

    # ── Load model ──
    print("Loading model...")
    model, _ = load_hmr_bilstm(RLSTM_CKPT, device)
    model.eval()

    # ── Load test data ──
    print(f"Loading test data: {INTER_TEST}")
    test   = np.load(INTER_TEST)
    X_test = test["X"].astype(np.float32)
    y_test = test["y"].astype(np.int64)

    # Global bounds for clamping (from normalized training distribution)
    data_min = float(X_test.min())
    data_max = float(X_test.max())
    print(f"  Data range: [{data_min:.3f}, {data_max:.3f}]")

    # ── Stratified subset (only correctly classified samples) ──
    print("Getting clean predictions...")
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 256):
            b = torch.from_numpy(X_test[i:i+256]).to(device)
            all_preds.append(model(b).argmax(-1).cpu().numpy())
    preds_clean = np.concatenate(all_preds)

    correct_mask = preds_clean == y_test
    X_correct = X_test[correct_mask]
    y_correct = y_test[correct_mask]
    X_sub, y_sub, sub_idx = select_stratified_subset(
        X_correct, y_correct, n_eval, rng
    )
    print(f"  Attacking {len(X_sub)} correctly classified samples (stratified)")

    # ── Run C&W attack in batches ──
    print(f"Running C&W attack (c={cw_c}, steps={cw_steps})...")
    batch_sz = 32   # smaller batch for C&W (inner optimization loop)
    X_adv_list = []
    for i in range(0, len(X_sub), batch_sz):
        bx = torch.from_numpy(X_sub[i:i+batch_sz])
        by = torch.from_numpy(y_sub[i:i+batch_sz]).long()
        x_adv = cw_attack_batch(
            model, bx, by, c=cw_c, lr=cw_lr, max_steps=cw_steps,
            kappa=kappa, data_min=data_min, data_max=data_max, device=device
        )
        X_adv_list.append(x_adv.cpu().numpy())
        print(f"  [{i+len(bx)}/{len(X_sub)}] done")

    X_adv = np.concatenate(X_adv_list, axis=0)

    # ── Evaluate on adversarial examples ──
    print("Evaluating adversarial examples...")
    adv_preds = []
    with torch.no_grad():
        for i in range(0, len(X_adv), 256):
            b = torch.from_numpy(X_adv[i:i+256]).to(device)
            adv_preds.append(model(b).argmax(-1).cpu().numpy())
    adv_preds = np.concatenate(adv_preds)

    # Attack Success Rate (ASR)
    asr_total = float((adv_preds != y_sub).mean())

    # Per-class ASR
    asr_per_class = {}
    l2_norms_per_class = {}
    for c in range(5):
        mask = y_sub == c
        if mask.sum() > 0:
            cn = CLASS_NAMES[c]
            asr_per_class[cn] = float((adv_preds[mask] != y_sub[mask]).mean())
            l2 = np.sqrt(((X_adv[mask] - X_sub[mask]) ** 2).sum(axis=(1, 2)))
            l2_norms_per_class[cn] = l2.tolist()
            print(f"  {cn}: ASR={asr_per_class[cn]:.4f}  "
                  f"L2_mean={l2.mean():.4f}  n={mask.sum()}")

    # Overall L2 norm
    l2_all = np.sqrt(((X_adv - X_sub) ** 2).sum(axis=(1, 2)))
    print(f"\n  Total ASR: {asr_total:.4f}")
    print(f"  Mean L2 norm: {l2_all.mean():.4f}  (std={l2_all.std():.4f})")

    # Adversarial F1
    adv_f1 = f1_score(y_sub, adv_preds, average="macro", zero_division=0)
    clean_f1 = f1_score(y_sub, preds_clean[correct_mask][sub_idx],
                        average="macro", zero_division=0)
    print(f"  Clean Macro F1: {clean_f1:.4f}  →  Adv Macro F1: {adv_f1:.4f}")

    # ── Plots ──
    print("\nGenerating plots...")
    plot_perturbation_norms(l2_norms_per_class,
                             paths["out_robust"] / "cw_perturbation_norms.png")
    plot_asr_by_class(asr_per_class,
                       paths["out_robust"] / "cw_asr_by_class.png")

    # ── Save results JSON ──
    results = {
        "experiment_version": cfg["experiment"]["version"],
        "run_id":             run_id,
        "checkpoint_hash":    get_checkpoint_hash(RLSTM_CKPT),
        "module":             "robustness_cw",
        "timestamp":          datetime.now().isoformat(),
        "config": {
            "cw_c":       cw_c,
            "cw_steps":   cw_steps,
            "cw_lr":      cw_lr,
            "kappa":      kappa,
            "n_eval":     n_eval,
            "batch_size": batch_sz,
            "data_range": [round(data_min, 4), round(data_max, 4)],
            "note":       "Only correctly-classified samples attacked (upper bound ASR)",
        },
        "metrics": {
            "asr_total":       round(asr_total, 4),
            "asr_per_class":   {k: round(v, 4) for k, v in asr_per_class.items()},
            "l2_mean":         round(float(l2_all.mean()), 4),
            "l2_std":          round(float(l2_all.std()),  4),
            "clean_f1_macro":  round(clean_f1, 4),
            "adv_f1_macro":    round(adv_f1,   4),
            "f1_drop":         round(clean_f1 - adv_f1, 4),
        }
    }

    out_path = paths["out_robust"] / "cw_attack_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  [OK] {out_path}")

    print("\n[T6 C&W Attack] Complete.")
    print(f"  ASR: {asr_total:.4f}  |  F1 drop: {clean_f1 - adv_f1:.4f}")


if __name__ == "__main__":
    main()
