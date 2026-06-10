"""
evaluate_pgd.py
---------------
Projected Gradient Descent (PGD) adversarial attack evaluation.
PGD = iterated FGSM with random restarts, clamped to an L-inf ball of radius eps.

Reference: Madry et al., "Towards Deep Learning Models Resistant to Adversarial
Attacks", ICLR 2018.

Usage (from d:/rlstm_final/):
    python evaluate_pgd.py
    python evaluate_pgd.py --epsilons 0.01 0.02 0.05 --steps 20 --alpha 0.005

Outputs (in results/figures/):
    pgd_results.csv
    pgd_baseline_comparison.csv
    pgd_vs_fgsm_comparison.png
    pgd_baseline_f1_vs_epsilon.png
    pgd_per_class_recall_eps0.02.png
"""
# Thêm vào đầu file:
from train import RLSTMLoss
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from report_results import load_hmr_bilstm
from evaluate_fgsm import load_baseline_model, build_test_loader, CLASS_NAMES, CLINICAL_CLASSES

# ---------------------------------------------------------------------------
# PGD attack
# ---------------------------------------------------------------------------

def pgd_attack(model, x, y, epsilon, alpha, steps, criterion, random_start=True, data_min=None, data_max=None):
    x_adv = x.clone().detach()
    
    if data_min is None: data_min = x.min()
    if data_max is None: data_max = x.max()

    if random_start:
        delta_init = torch.zeros_like(x).uniform_(-epsilon, epsilon)
        x_adv = (x + delta_init).clamp(data_min, data_max).detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        outputs = model(x_adv)
        loss, _ = criterion(outputs, y, r_fwd=None, r_bwd=None)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
            x_adv = (x + delta).clamp(data_min, data_max).detach()

    return x_adv, (x_adv - x).detach()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model_pgd(model, loader, device, epsilon, alpha, steps, criterion, data_min, data_max):
    model.eval()

    all_orig_preds = []
    all_preds      = []
    all_labels     = []
    orig_correct_list = []
    adv_wrong_list    = []
    orig_conf_list    = []
    adv_conf_list     = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Clean examples
        with torch.no_grad():
            orig_logits = model(x)
            orig_preds  = orig_logits.argmax(dim=1)
            orig_probs  = F.softmax(orig_logits, dim=1)
            orig_conf_list.append(orig_probs.max(dim=1).values.cpu().numpy())

        # Adversarial examples (PGD)
        if epsilon == 0.0:
            x_adv = x.clone().detach()
        else:
            x_adv, _ = pgd_attack(model, x, y, epsilon, alpha, steps, criterion, data_min=data_min, data_max=data_max)

        with torch.no_grad():
            adv_logits = model(x_adv)
            adv_preds  = adv_logits.argmax(dim=1)
            adv_probs  = F.softmax(adv_logits, dim=1)
            adv_conf_list.append(adv_probs.max(dim=1).values.cpu().numpy())

        all_orig_preds.extend(orig_preds.cpu().numpy())
        all_preds.extend(adv_preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        orig_correct_list.append((orig_preds == y).cpu().numpy())
        adv_wrong_list.append((adv_preds != y).cpu().numpy())

    all_orig_preds = np.array(all_orig_preds)
    all_preds      = np.array(all_preds)
    all_labels     = np.array(all_labels)
    orig_correct   = np.concatenate(orig_correct_list)
    adv_wrong      = np.concatenate(adv_wrong_list)
    orig_conf      = np.concatenate(orig_conf_list)
    adv_conf       = np.concatenate(adv_conf_list)

    macro_f1     = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    asr = float((orig_correct & adv_wrong).sum() / orig_correct.sum()) if orig_correct.sum() > 0 else 0.0

    per_clean, per_adv = {}, {}
    for cls in CLINICAL_CLASSES:
        name = CLASS_NAMES[cls]
        mask = all_labels == cls
        per_clean[f"recall_clean_{name}"] = float((all_orig_preds[mask] == cls).sum() / mask.sum()) if mask.sum() > 0 else 0.0
        per_adv[f"recall_adv_{name}"]     = float((all_preds[mask] == cls).sum() / mask.sum())      if mask.sum() > 0 else 0.0

    result = {
        "epsilon": epsilon, "alpha": alpha, "steps": steps,
        "macro_f1": float(macro_f1),
        "macro_recall": float(macro_recall),
        "attack_success_rate": asr,
        "avg_confidence": float(adv_conf.mean()),
        "orig_avg_confidence": float(orig_conf.mean()),
        "confidence_drop": float((orig_conf - adv_conf).mean()),
    }
    result.update(per_clean)
    result.update(per_adv)
    return result


def evaluate_pgd_grid(model, test_loader, device, criterion, epsilons, alpha, steps):
    results = []
    
    # Calculate global data min/max for clamping
    # We can get this directly from the loader
    all_x = []
    for x, _ in test_loader:
        all_x.append(x)
    all_x = torch.cat(all_x)
    data_min = float(all_x.min())
    data_max = float(all_x.max())
    print(f"Global dataset min: {data_min:.4f}, max: {data_max:.4f} for PGD clamping")

    for epsilon in epsilons:
        print(f"  PGD epsilon={epsilon:.3f}  alpha={alpha:.4f}  steps={steps}")
        r = evaluate_model_pgd(model, test_loader, device, epsilon, alpha, steps, criterion, data_min, data_max)
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def compare_models_pgd(models_dict, epsilons, alpha, steps, device, test_loader):
    # Tạo criterion giống lúc train
    cw_path = Path("data/processed/class_weights.npy")
    if cw_path.exists():
        class_weights = torch.from_numpy(np.load(cw_path)).float().to(device)
    else:
        class_weights = None
    
    criterion = RLSTMLoss(
        lambda_smooth=0.003,
        class_weights=class_weights,
        use_focal=True,
        focal_gamma=1.5,
    )
    print("Using FocalLoss (gamma=1.5) with class weights for PGD attack")

    comparison = {}
    for model_name, ckpt in models_dict.items():
        if not Path(ckpt).exists():
            print(f"  [SKIP] {model_name}: checkpoint not found ({ckpt})")
            continue
        print(f"\n[PGD] Evaluating {model_name}")
        try:
            if model_name == "HMR-BiLSTM":
                model, _ = load_hmr_bilstm(ckpt, device)
            else:
                model, _ = load_baseline_model(ckpt, device)
            comparison[model_name] = evaluate_pgd_grid(
                model, test_loader, device, criterion, epsilons, alpha, steps
            )
            print(f"  [OK] {model_name} done")
        except Exception as e:
            print(f"  [ERR] {model_name}: {e}")
    return comparison


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

COLORS  = {"LSTM": "#1f77b4", "BiLSTM": "#ff7f0e", "HMR-BiLSTM": "#2ca02c"}
MARKERS = {"LSTM": "o",       "BiLSTM": "s",        "HMR-BiLSTM": "^"}


def plot_pgd_f1_vs_epsilon(comparison, output_dir):
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, results in comparison.items():
        eps = [r["epsilon"] for r in results]
        f1  = [r["macro_f1"] for r in results]
        ax.plot(eps, f1,
                marker=MARKERS.get(model_name, "x"),
                color=COLORS.get(model_name, None),
                label=model_name, linewidth=2.5, markersize=8)
    ax.set_xlabel("Epsilon (L-inf radius)", fontsize=12)
    ax.set_ylabel("Macro F1", fontsize=12)
    ax.set_title("PGD Robustness: Macro F1 vs Epsilon", fontsize=13, fontweight="bold")
    ax.set_ylim([0.5, 1.0])
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=11)
    plt.tight_layout()
    path = output_dir / "pgd_baseline_f1_vs_epsilon.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[FIG] {path}")


def plot_pgd_vs_fgsm(fgsm_csv, pgd_comparison, output_dir):
    """Side-by-side bar chart: FGSM F1 vs PGD F1 at eps=0.02."""
    # Load FGSM numbers from existing CSV
    fgsm_data = {}
    try:
        with open(fgsm_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fgsm_data[row["model"]] = float(row["fgsm_f1"])
    except Exception:
        pass   # if CSV missing, skip FGSM bars

    TARGET_EPS = 0.02
    models = list(pgd_comparison.keys())
    x = np.arange(len(models))
    w = 0.28

    pgd_f1 = []
    for m in models:
        # find result closest to TARGET_EPS
        res = min(pgd_comparison[m], key=lambda r: abs(r["epsilon"] - TARGET_EPS))
        pgd_f1.append(res["macro_f1"])

    fgsm_f1 = [fgsm_data.get(m, None) for m in models]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, [v if v else 0 for v in fgsm_f1], width=w,
           color=[COLORS.get(m, "#999") for m in models], alpha=0.65, label="FGSM")
    ax.bar(x + w/2, pgd_f1, width=w,
           color=[COLORS.get(m, "#999") for m in models], alpha=0.95, label="PGD", hatch="//")

    for i, (f, p) in enumerate(zip(fgsm_f1, pgd_f1)):
        if f: ax.text(i - w/2, f + 0.005, f"{f:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, p + 0.005, f"{p:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Macro F1", fontsize=12)
    ax.set_title(f"FGSM vs PGD Macro F1 at epsilon={TARGET_EPS}", fontsize=13, fontweight="bold")
    ax.set_ylim([0.55, 1.0])
    ax.grid(alpha=0.3, linestyle="--", axis="y")
    ax.legend(fontsize=11)
    plt.tight_layout()
    path = output_dir / "pgd_vs_fgsm_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[FIG] {path}")


def plot_pgd_per_class_recall(comparison, output_dir, target_eps=0.02):
    """Grouped bar chart of per-class recall (clean vs PGD) for S, V, F."""
    clinical = ["S", "V", "F"]
    models   = list(comparison.keys())
    n_models = len(models)
    n_cls    = len(clinical)
    w = 0.12
    x = np.arange(n_cls)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model_name in enumerate(models):
        res = min(comparison[model_name], key=lambda r: abs(r["epsilon"] - target_eps))
        c_vals = [res.get(f"recall_clean_{c}", 0) for c in clinical]
        a_vals = [res.get(f"recall_adv_{c}",   0) for c in clinical]
        offset_c = (i - n_models/2 + 0.25) * w * 2
        offset_a = offset_c + w

        ax.bar(x + offset_c, c_vals, width=w, color=COLORS.get(model_name, "#999"),
               alpha=0.9, label=f"{model_name} clean" if i == 0 else "_nolegend_")
        ax.bar(x + offset_a, a_vals, width=w, color=COLORS.get(model_name, "#999"),
               alpha=0.45, hatch="//",
               label=f"{model_name} PGD" if i == 0 else "_nolegend_")

    # Build proper legend entries
    from matplotlib.patches import Patch
    legend_handles = []
    for m in models:
        legend_handles.append(Patch(facecolor=COLORS.get(m, "#999"), alpha=0.9,  label=f"{m} (clean)"))
        legend_handles.append(Patch(facecolor=COLORS.get(m, "#999"), alpha=0.45, hatch="//", label=f"{m} (PGD)"))
    ax.legend(handles=legend_handles, fontsize=8, ncol=2, loc="lower left")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Class {c}" for c in clinical], fontsize=12)
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_ylim([0.0, 1.05])
    ax.set_title(f"Per-class Recall: Clean vs PGD (epsilon={target_eps})", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, linestyle="--", axis="y")
    plt.tight_layout()
    path = output_dir / f"pgd_per_class_recall_eps{target_eps:.2f}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[FIG] {path}")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_pgd_csv(comparison, output_dir):
    rows = []
    for model_name, results in comparison.items():
        # summary row at eps=0.02
        r02 = min(results, key=lambda r: abs(r["epsilon"] - 0.02))
        r05 = min(results, key=lambda r: abs(r["epsilon"] - 0.05))
        rows.append({
            "model":       model_name,
            "clean_f1":    f"{results[0]['macro_f1']:.4f}",   # eps=0
            "pgd_f1_002":  f"{r02['macro_f1']:.4f}",
            "pgd_f1_005":  f"{r05['macro_f1']:.4f}",
            "f1_drop_002": f"{results[0]['macro_f1'] - r02['macro_f1']:.4f}",
            "asr_002":     f"{r02['attack_success_rate']:.4f}",
            "asr_005":     f"{r05['attack_success_rate']:.4f}",
        })

    csv_path = Path("results/tables/pgd_baseline_comparison.csv"); csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] {csv_path}")

    # also dump full JSON
    json_path = Path("results/logs/pgd_baseline_comparison.json"); json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"[JSON] {json_path}")

    return csv_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser(description="PGD adversarial robustness evaluation")
    parser.add_argument("--epsilons", nargs="*", type=float,
                        default=[0.0, 0.01, 0.02, 0.05])
    parser.add_argument("--steps",   type=int,   default=20,
                        help="PGD iterations (default: 20)")
    parser.add_argument("--alpha",   type=float, default=0.005,
                        help="Step size per PGD iteration (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device : {device}")
    print(f"PGD    : steps={args.steps}  alpha={args.alpha}  epsilons={args.epsilons}")

    test_loader = build_test_loader(batch_size=args.batch_size)

    models_dict = {
        "LSTM":       "results/checkpoints/best_lstm.pt",
        "BiLSTM":     "results/checkpoints/best_bilstm.pt",
        "HMR-BiLSTM": "results/checkpoints/inter_best_rlstm.pt",
    }

    comparison = compare_models_pgd(
        models_dict, args.epsilons, args.alpha, args.steps, device, test_loader
    )

    if not comparison:
        print("No models evaluated. Exiting.")
        return

    # --- Save CSV / JSON ---
    csv_path = save_pgd_csv(comparison, output_dir)

    # --- Figures ---
    plot_pgd_f1_vs_epsilon(comparison, output_dir)
    plot_pgd_vs_fgsm(
        output_dir / "fgsm_baseline_summary.csv",
        comparison, output_dir
    )
    plot_pgd_per_class_recall(comparison, output_dir, target_eps=0.02)

    # --- Console summary ---
    print("\n" + "=" * 62)
    print(" PGD RESULTS SUMMARY  (eps=0.02, steps={})".format(args.steps))
    print("=" * 62)
    print(f"  {'Model':<14} {'F1-clean':>9} {'F1-PGD':>9} {'F1-drop':>9} {'ASR':>9}")
    print("  " + "-" * 46)
    for model_name, results in comparison.items():
        clean = results[0]["macro_f1"]
        r02   = min(results, key=lambda r: abs(r["epsilon"] - 0.02))
        drop  = clean - r02["macro_f1"]
        asr   = r02["attack_success_rate"]
        print(f"  {model_name:<14} {clean:>9.4f} {r02['macro_f1']:>9.4f} {drop:>9.4f} {asr:>9.4f}")
    print("=" * 62)
    print("\nDone. Results in:", output_dir)


if __name__ == "__main__":
    main()
