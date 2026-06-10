"""
T3 — Integrated Gradients for Trustworthy ECG Classification (captum).

Annotates QRS, P-wave, T-wave regions on ECG signal.
Uses zero-signal as baseline.

Outputs:
  outputs/<run_id>/explainability/
    ig_normal.png
    ig_pvc.png      (class V)
    ig_apc.png      (class S)
    ig_fusion.png   (class F)
    ig_results.json (merged into explainability/results.json by T8)
"""

import json
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

from configs.paths import get_run_id, build_paths, RLSTM_CKPT, get_checkpoint_hash, INTER_TEST
from report_results import load_hmr_bilstm


CLASS_NAMES   = {0: "N (Normal)", 1: "S (APC)", 2: "V (PVC)", 3: "F (Fusion)"}
PLOT_CLASSES  = {0: "ig_normal.png", 1: "ig_apc.png", 2: "ig_pvc.png", 3: "ig_fusion.png"}

# Approximate ECG region boundaries for MIT-BIH 187-sample beat
# These are rough heuristics; the plot just annotates regions visually
ECG_REGIONS = {
    "P-wave":  (10,  40),
    "QRS":     (60,  100),
    "T-wave":  (110, 155),
}


def select_one_sample_per_class(X, y, preds, cls, rng):
    """Pick one correctly classified sample for the class, or any true class sample."""
    correct = np.where((y == cls) & (preds == cls))[0]
    if len(correct) > 0:
        idx = rng.choice(correct)
    else:
        fallback = np.where(y == cls)[0]
        if len(fallback) == 0:
            return None
        idx = rng.choice(fallback)
    return int(idx)


def plot_ig(signal, attribution, cls_name, save_path, title):
    """Overlay integrated gradients on ECG signal with anatomical region annotations."""
    T   = len(signal)
    t   = np.arange(T)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                             gridspec_kw={'height_ratios': [2.5, 1]})
    fig.suptitle(title, fontsize=14, fontweight='bold')

    ax1 = axes[0]
    ax1.plot(t, signal, color='#1565C0', linewidth=1.5, label='ECG signal', zorder=3)

    # Region annotations
    colors_region = {'P-wave': '#FFF9C4', 'QRS': '#FFCCBC', 'T-wave': '#C8E6C9'}
    for region, (s, e) in ECG_REGIONS.items():
        ax1.axvspan(s, e, alpha=0.35, color=colors_region[region], label=region, zorder=1)

    # IG overlay as coloured fill
    pos = np.where(attribution >= 0, attribution, 0)
    neg = np.where(attribution < 0, np.abs(attribution), 0)
    ax1.fill_between(t, signal - pos * 0.5, signal + pos * 0.5,
                     alpha=0.5, color='#D32F2F', label='Positive IG', zorder=2)
    ax1.fill_between(t, signal - neg * 0.5, signal + neg * 0.5,
                     alpha=0.5, color='#388E3C', label='Negative IG', zorder=2)

    ax1.set_ylabel("Amplitude", fontsize=11)
    ax1.legend(loc='upper right', fontsize=8, ncol=3)
    ax1.grid(alpha=0.25)
    ax1.set_xlim([0, T - 1])

    # Bottom panel: attribution bar chart
    ax2 = axes[1]
    ax2.bar(t, attribution, color=np.where(attribution >= 0, '#D32F2F', '#388E3C'),
            width=0.9, alpha=0.8)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xlabel("Time Step", fontsize=11)
    ax2.set_ylabel("Attribution", fontsize=10)
    ax2.set_xlim([0, T - 1])
    ax2.grid(alpha=0.2, linestyle='--', axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {save_path}")


def jaccard_with_tolerance(set_a, set_b, tolerance=2, T=187):
    """Symmetric Jaccard index with a temporal tolerance window."""
    matched_a = set(a for a in set_a if any(abs(a - b) <= tolerance for b in set_b))
    matched_b = set(b for b in set_b if any(abs(b - a) <= tolerance for a in set_a))
    intersection = (len(matched_a) + len(matched_b)) / 2.0
    union = len(set_a) + len(set_b) - intersection
    return intersection / max(1.0, union)


def main():
    config_path = Path("configs/experiment_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_id = get_run_id(cfg)
    paths  = build_paths(run_id)
    paths["out_explain"].mkdir(parents=True, exist_ok=True)

    seed = cfg.get("seed", 42)
    rng  = np.random.default_rng(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Run ID: {run_id}")

    # Load model
    print("Loading model...")
    model, _ = load_hmr_bilstm(RLSTM_CKPT, device)
    model.eval()

    # Load test data & get predictions (using centralized INTER_TEST path)
    print("Loading test data...")
    print(f"  Using: {INTER_TEST}")
    test   = np.load(INTER_TEST)
    X_test = test["X"].astype(np.float32)
    y_test = test["y"].astype(np.int64)

    all_preds = []
    X_t = torch.from_numpy(X_test)
    with torch.no_grad():
        for i in range(0, len(X_t), 256):
            b = X_t[i:i+256].to(device)
            all_preds.append(model(b).argmax(dim=-1).cpu().numpy())
    preds_all = np.concatenate(all_preds)

    # Integrated Gradients — one sample per class (0-3)
    ig = IntegratedGradients(model)
    n_steps = 50
    ig_stats = {}

    for cls, fname in PLOT_CLASSES.items():
        cls_label = CLASS_NAMES[cls]
        print(f"Computing IG for class {cls_label}...")

        idx = select_one_sample_per_class(X_test, y_test, preds_all, cls, rng)
        if idx is None:
            print(f"  Warning: no sample found for class {cls_label}, skipping.")
            continue

        x = torch.from_numpy(X_test[idx:idx+1]).to(device)   # (1, T, 1)
        baseline = torch.zeros_like(x)

        # Compute IG attributions for target class
        attrs, delta = ig.attribute(
            x, baseline, target=cls,
            n_steps=n_steps, return_convergence_delta=True
        )
        attribution = attrs.squeeze().cpu().detach().numpy()   # (T,) or (T,1)
        if attribution.ndim > 1:
            attribution = attribution.squeeze(-1)
        signal = X_test[idx].squeeze()                        # (T,)

        pred_lbl = CLASS_NAMES.get(int(preds_all[idx]), str(preds_all[idx]))
        true_lbl = CLASS_NAMES[cls]
        title = (f"Integrated Gradients — {cls_label}\n"
                 f"True: {true_lbl}  |  Predicted: {pred_lbl}  |  "
                 f"Convergence delta: {delta.item():.4f}")

        plot_ig(signal, attribution, cls_label, paths["out_explain"] / fname, title)

        # Store stats
        top10 = np.argsort(np.abs(attribution))[::-1][:10].tolist()
        ig_stats[CLASS_NAMES[cls].split()[0]] = {
            "top10_timesteps_ig": top10,
            "convergence_delta": float(delta.item()),
            "max_abs_attr": float(np.abs(attribution).max())
        }

    # Save IG-specific JSON
    ig_json = {
        "experiment_version": cfg["experiment"]["version"],
        "run_id": run_id,
        "checkpoint_hash": get_checkpoint_hash(RLSTM_CKPT),
        "module": "explainability_ig",
        "timestamp": datetime.now().isoformat(),
        "metrics": ig_stats
    }
    ig_json_path = paths["out_explain"] / "ig_results.json"
    with open(ig_json_path, "w", encoding="utf-8") as f:
        json.dump(ig_json, f, indent=2)
    print(f"  [OK] ig_results.json")

    # Compute Jaccard Similarity with SHAP (if available)
    shap_results_path = paths["out_explain"] / "results.json"
    if shap_results_path.exists():
        with open(shap_results_path, "r", encoding="utf-8") as f:
            shap_results = json.load(f)
        
        consistency = {}
        for cls_name, ig_data in ig_stats.items():
            if cls_name in shap_results.get("metrics", {}).get("per_class", {}):
                shap_top10 = set(shap_results["metrics"]["per_class"][cls_name].get("top10_timesteps", []))
                ig_top10 = set(ig_data["top10_timesteps_ig"])
                
                if shap_top10 and ig_top10:
                    jaccard = jaccard_with_tolerance(shap_top10, ig_top10, tolerance=2)
                    consistency[cls_name] = float(jaccard)
                    print(f"  SHAP vs IG Jaccard Similarity with tolerance ({cls_name}): {jaccard:.4f}")
        
        if consistency:
            consistency_path = paths["out_explain"] / "shap_ig_consistency.json"
            with open(consistency_path, "w", encoding="utf-8") as f:
                json.dump({"jaccard_similarity": consistency}, f, indent=2)
            print(f"  [OK] Saved consistency to {consistency_path}")

    print("\nIntegrated Gradients analysis completed.")


if __name__ == "__main__":
    main()
