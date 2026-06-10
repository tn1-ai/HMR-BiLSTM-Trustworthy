"""
T7 — AutoAttack (Parameter-free Adversarial Evaluation).

AutoAttack (Croce & Hein, 2020) is the standard benchmark for reliable
adversarial robustness evaluation. It combines four complementary attacks:
  1. APGD-CE:   adaptive PGD with cross-entropy (gradient-based)
  2. APGD-DLR:  adaptive PGD with DLR loss (gradient-based, margin-based)
  3. FAB:       Fast Adaptive Boundary (decision-based, no gradient needed)
  4. Square:    Square Attack (score-based, black-box)

Using all 4 ensures we detect gradient masking:
  - If APGD (white-box) ASR << Square (black-box) ASR → gradient masking confirmed
  - AutoAttack standard version uses all 4 sequentially

Why this answers P3 (gradient masking concern):
  If PGD-ASR ≈ 1% but AutoAttack-ASR >> 1%, we CONFIRM gradient masking.
  The paper should then say: "Our low PGD-ASR reflects gradient masking.
  AutoAttack ASR=[X]% gives the true lower bound on adversarial accuracy."

Using `autoattack` package (pip install autoattack).
Note: autoattack operates on [0,1] normalized inputs by default.
We normalize our data to [0,1] for AutoAttack and denormalize after.

Outputs:
  outputs/<run_id>/robustness/
    autoattack_results.json
    autoattack_comparison.png   ← PGD vs AutoAttack ASR comparison
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
from sklearn.metrics import f1_score, accuracy_score

from configs.paths import (
    get_run_id, build_paths, RLSTM_CKPT, get_checkpoint_hash, INTER_TEST
)
from report_results import load_hmr_bilstm
from robustness.cw_attack import select_stratified_subset


CLASS_NAMES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}


# ── [0,1] normalization wrapper ───────────────────────────────────────────────

class NormalizedModelWrapper(nn.Module):
    """
    Wrapper that maps AutoAttack's [0,1] input range back to the
    model's expected normalized ECG range [data_min, data_max].

    AutoAttack internally generates perturbations in [0,1] space.
    We need to map those back to Z-score space for our model.
    """
    def __init__(self, model: nn.Module, data_min: float, data_max: float):
        super().__init__()
        self.model    = model
        self.data_min = data_min
        self.data_max = data_max

    def forward(self, x):
        # x is in [0, 1]; map back to [data_min, data_max]
        x_real = x * (self.data_max - self.data_min) + self.data_min
        return self.model(x_real)


def to_unit_range(X: np.ndarray, data_min: float, data_max: float) -> np.ndarray:
    """Map X from [data_min, data_max] → [0, 1] for AutoAttack."""
    return (X - data_min) / (data_max - data_min + 1e-8)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_attack_comparison(pgd_asr: float, cw_asr: float, aa_asr: float,
                            out_path: Path):
    """Bar chart comparing FGSM/PGD/C&W/AutoAttack ASRs."""
    labels = ["PGD-20\n(whitebox)", "C&W L2\n(whitebox)", "AutoAttack\n(ensemble)"]
    values = [pgd_asr * 100, cw_asr * 100, aa_asr * 100]

    colors = ["#1565C0", "#E65100", "#B71C1C"]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=12,
                fontweight="bold")

    ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
    ax.set_title("Adversarial Robustness: Attack Comparison\n"
                 "(higher gap PGD↔AutoAttack → gradient masking)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim([0, max(values) * 1.25 + 5])
    ax.grid(alpha=0.25, linestyle="--", axis="y")

    # Annotate masking gap
    gap = aa_asr - pgd_asr
    if gap > 0.05:
        ax.annotate(
            f"Masking gap: +{gap*100:.1f}%",
            xy=(2, aa_asr * 100),
            xytext=(1.5, aa_asr * 100 + max(values) * 0.1),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=10, color="#B71C1C", fontweight="bold"
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out_path.name}")


# ── Quick PGD baseline (for comparison in same run) ──────────────────────────

def pgd_attack(model, x, y, epsilon, alpha, steps, data_min, data_max, device):
    """PGD with global clamping — same protocol as evaluate_pgd.py."""
    model.eval()
    x_adv = x.clone().detach()
    # Random init
    x_adv = x_adv + torch.zeros_like(x_adv).uniform_(-epsilon, epsilon)
    x_adv = x_adv.clamp(data_min, data_max).detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            delta = (x_adv - x).clamp(-epsilon, epsilon)
            x_adv = (x + delta).clamp(data_min, data_max).detach()
    return x_adv


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    config_path = Path("configs/experiment_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_id = get_run_id(cfg)
    paths  = build_paths(run_id)
    paths["out_robust"].mkdir(parents=True, exist_ok=True)

    rob_cfg      = cfg["robustness"]
    aa_eps       = rob_cfg.get("autoattack_eps",  0.02)
    aa_norm      = rob_cfg.get("autoattack_norm", "Linf")
    pgd_eps      = 0.02
    seed         = cfg.get("seed", 42)

    n_eval = 200   # subset to attack

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Run ID: {run_id}")
    print(f"AutoAttack ε={aa_eps} norm={aa_norm}")

    # ── Load model ──
    print("Loading model...")
    model, _ = load_hmr_bilstm(RLSTM_CKPT, device)
    model.eval()

    # ── Load test data ──
    print(f"Loading test data: {INTER_TEST}")
    test   = np.load(INTER_TEST)
    X_test = test["X"].astype(np.float32)
    y_test = test["y"].astype(np.int64)

    data_min = float(X_test.min())
    data_max = float(X_test.max())
    print(f"  Data range: [{data_min:.3f}, {data_max:.3f}]")

    # ── Get clean predictions; keep only correctly classified ──
    print("Getting clean predictions...")
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 256):
            b = torch.from_numpy(X_test[i:i+256]).to(device)
            all_preds.append(model(b).argmax(-1).cpu().numpy())
    preds_clean = np.concatenate(all_preds)

    correct_mask = preds_clean == y_test
    X_correct    = X_test[correct_mask]
    y_correct    = y_test[correct_mask]
    X_sub, y_sub, _ = select_stratified_subset(X_correct, y_correct, n_eval, rng)
    print(f"  Subset: {len(X_sub)} correctly classified samples (stratified)")

    clean_acc = accuracy_score(y_sub, preds_clean[correct_mask][:len(y_sub)])
    clean_f1  = f1_score(y_sub, preds_clean[correct_mask][:len(y_sub)],
                          average="macro", zero_division=0)

    # ── PGD-20 baseline (for comparison) ──
    print("Running PGD-20 baseline...")
    X_pgd_list = []
    pgd_batch = 64
    for i in range(0, len(X_sub), pgd_batch):
        bx = torch.from_numpy(X_sub[i:i+pgd_batch]).to(device)
        by = torch.from_numpy(y_sub[i:i+pgd_batch]).long().to(device)
        x_pgd = pgd_attack(model, bx, by, pgd_eps, pgd_eps/4, 20,
                            data_min, data_max, device)
        X_pgd_list.append(x_pgd.cpu().numpy())
    X_pgd = np.concatenate(X_pgd_list, axis=0)

    pgd_preds = []
    with torch.no_grad():
        for i in range(0, len(X_pgd), 256):
            b = torch.from_numpy(X_pgd[i:i+256]).to(device)
            pgd_preds.append(model(b).argmax(-1).cpu().numpy())
    pgd_preds = np.concatenate(pgd_preds)
    pgd_asr   = float((pgd_preds != y_sub).mean())
    pgd_f1    = f1_score(y_sub, pgd_preds, average="macro", zero_division=0)
    print(f"  PGD-20 ASR: {pgd_asr:.4f}  Macro F1: {pgd_f1:.4f}")

    # ── AutoAttack ──
    print(f"Running AutoAttack ({aa_norm}, ε={aa_eps})...")
    # AutoAttack expects input in [0,1], output logits
    X_sub_unit  = to_unit_range(X_sub, data_min, data_max)
    x_aa_tensor = torch.from_numpy(X_sub_unit).to(device)      # (N, T, 1)
    y_aa_tensor = torch.from_numpy(y_sub).long().to(device)

    # Transpose to (N, 1, T) — AutoAttack expects (N, C, H) or (N, C, L)
    x_aa_tensor = x_aa_tensor.permute(0, 2, 1)   # (N, 1, T)

    # Wrapper: receives (N, 1, T), permutes back, runs model
    class AAWrapper(nn.Module):
        def __init__(self, m, d_min, d_max):
            super().__init__()
            self.m = m
            self.d_min = d_min
            self.d_max = d_max
        def forward(self, x):
            # x: (N, 1, T) in [0,1] → (N, T, 1) in Z-score range
            x = x.permute(0, 2, 1)
            x_real = x * (self.d_max - self.d_min) + self.d_min
            return self.m(x_real)

    aa_wrapper = AAWrapper(model, data_min, data_max).to(device)
    aa_wrapper.eval()

    try:
        from autoattack import AutoAttack
        adversary = AutoAttack(
            aa_wrapper, norm=aa_norm, eps=aa_eps,
            version="standard", device=device, verbose=True
        )
        # AutoAttack returns perturbed x_adv in [0,1], (N, 1, T)
        x_adv_aa = adversary.run_standard_evaluation(
            x_aa_tensor, y_aa_tensor, bs=32
        )
        aa_available = True
    except ImportError:
        print("  [!] autoattack package not installed. Using APGD approximation.")
        # Fallback: stronger PGD (100 steps) as AutoAttack approximation
        X_apgd_list = []
        for i in range(0, len(X_sub), 32):
            bx = torch.from_numpy(X_sub[i:i+32]).to(device)
            by = torch.from_numpy(y_sub[i:i+32]).long().to(device)
            x_apgd = pgd_attack(model, bx, by, pgd_eps, pgd_eps/10, 100,
                                 data_min, data_max, device)
            X_apgd_list.append(x_apgd.cpu().numpy())
        X_apgd = np.concatenate(X_apgd_list, axis=0)
        # Convert to unit format for uniformity
        x_adv_aa = torch.from_numpy(
            to_unit_range(X_apgd, data_min, data_max)
        ).permute(0, 2, 1).to(device)
        aa_available = False

    # Evaluate AutoAttack adversarial examples
    aa_preds = []
    with torch.no_grad():
        for i in range(0, x_adv_aa.size(0), 256):
            b = x_adv_aa[i:i+256].to(device)
            aa_preds.append(aa_wrapper(b).argmax(-1).cpu().numpy())
    aa_preds = np.concatenate(aa_preds)

    aa_asr  = float((aa_preds != y_sub).mean())
    aa_f1   = f1_score(y_sub, aa_preds, average="macro", zero_division=0)
    aa_f1pc = f1_score(y_sub, aa_preds, average=None, zero_division=0)
    print(f"  AutoAttack ASR: {aa_asr:.4f}  Macro F1: {aa_f1:.4f}")

    # Per-class ASR
    aa_asr_per_class = {}
    for c in range(5):
        mask = y_sub == c
        if mask.sum() > 0:
            cn = CLASS_NAMES[c]
            aa_asr_per_class[cn] = float((aa_preds[mask] != y_sub[mask]).mean())
            print(f"    {cn}: ASR={aa_asr_per_class[cn]:.4f}  n={mask.sum()}")

    # Gradient masking diagnosis
    masking_gap = aa_asr - pgd_asr
    gradient_masking_suspected = masking_gap > 0.15
    print(f"\n  Masking gap (AA - PGD): {masking_gap:.4f}")
    if gradient_masking_suspected:
        print("  ⚠ Gradient masking SUSPECTED (gap > 0.15)")
    else:
        print("  ✓ No strong gradient masking evidence")

    # ── Load C&W results for comparison (if available) ──
    cw_results_path = paths["out_robust"] / "cw_attack_results.json"
    cw_asr = 0.0
    if cw_results_path.exists():
        with open(cw_results_path) as f:
            cw_data = json.load(f)
        cw_asr = cw_data.get("metrics", {}).get("asr_total", 0.0)

    # ── Plots ──
    print("\nGenerating plots...")
    plot_attack_comparison(
        pgd_asr, cw_asr, aa_asr,
        paths["out_robust"] / "autoattack_comparison.png"
    )

    # ── Save results JSON ──
    results = {
        "experiment_version":    cfg["experiment"]["version"],
        "run_id":                run_id,
        "checkpoint_hash":       get_checkpoint_hash(RLSTM_CKPT),
        "module":                "robustness_autoattack",
        "timestamp":             datetime.now().isoformat(),
        "autoattack_available":  aa_available,
        "config": {
            "epsilon":      aa_eps,
            "norm":         aa_norm,
            "n_eval":       n_eval,
            "aa_version":   "standard" if aa_available else "pgd100_fallback",
            "data_range":   [round(data_min, 4), round(data_max, 4)],
        },
        "metrics": {
            "clean_accuracy":         round(clean_acc, 4),
            "clean_f1_macro":         round(clean_f1,  4),
            "pgd20_asr":              round(pgd_asr,   4),
            "pgd20_f1_macro":         round(pgd_f1,    4),
            "autoattack_asr":         round(aa_asr,    4),
            "autoattack_f1_macro":    round(aa_f1,     4),
            "autoattack_f1_per_class":{CLASS_NAMES[i]: round(float(aa_f1pc[i]), 4)
                                       for i in range(len(aa_f1pc))},
            "autoattack_asr_per_class": {k: round(v, 4)
                                          for k, v in aa_asr_per_class.items()},
            "masking_gap_aa_minus_pgd": round(masking_gap, 4),
            "gradient_masking_suspected": gradient_masking_suspected,
        },
        "interpretation": {
            "gradient_masking_note": (
                "Masking gap = AutoAttack ASR - PGD-20 ASR. "
                "Gap > 0.15 → gradient masking suspected. "
                "Use this to honestly characterize robustness in paper."
            ),
            "paper_claim": (
                f"Under AutoAttack (ε={aa_eps}, {aa_norm}), ASR={aa_asr:.4f}. "
                f"PGD-20 ASR={pgd_asr:.4f}. "
                + ("Gradient masking suspected (Δ={:.4f}).".format(masking_gap)
                   if gradient_masking_suspected else
                   "No strong gradient masking detected.")
            )
        }
    }

    out_path = paths["out_robust"] / "autoattack_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  [OK] {out_path}")

    print("\n[T7 AutoAttack] Complete.")
    print(f"  PGD-20 ASR: {pgd_asr:.4f}  →  AutoAttack ASR: {aa_asr:.4f}")
    print(f"  Masking gap: {masking_gap:+.4f}  "
          + ("⚠ MASKING" if gradient_masking_suspected else "✓ clean"))


if __name__ == "__main__":
    main()
