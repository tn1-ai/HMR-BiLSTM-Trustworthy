"""
T2 — SHAP GradientExplainer for Trustworthy ECG Classification.

Uses GradientExplainer (vs DeepExplainer) because HMR-BiLSTM contains
LayerNorm layers that DeepExplainer cannot handle correctly.
GradientExplainer works with any differentiable PyTorch function.

Classes: N (0), S/APC (1), V/PVC (2), F/Fusion (3)
Sample strategy (per class):
  - 10 correctly predicted samples
  - 5 misclassified samples  -> "Why does the model fail?"

Outputs:
  outputs/<run_id>/explainability/
    shap_summary_plot.png
    shap_class_N.png  / S / V / F
    shap_misclassified_N.png  / S / V / F
    shap_importance_ranking.csv
    results.json
"""

import json
import csv
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from configs.paths import get_run_id, build_paths, RLSTM_CKPT, get_checkpoint_hash, INTER_TRAIN, INTER_TEST
from report_results import load_hmr_bilstm


CLASS_NAMES  = {0: "N", 1: "S", 2: "V", 3: "F"}
SHAP_CLASSES = [0, 1, 2, 3]


# ── Wrapper: logits only ─────────────────────────────────────────────────────
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


# ── Sample selection ─────────────────────────────────────────────────────────
def select_samples(X, labels, preds, cls, n_correct=10, n_mis=5, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    tp  = np.where((labels == cls) & (preds == cls))[0]
    fn  = np.where((labels == cls) & (preds != cls))[0]
    nc  = min(n_correct, len(tp))
    nm  = min(n_mis,     len(fn))
    ci  = rng.choice(tp, nc, replace=False) if nc > 0 else np.array([], dtype=int)
    mi  = rng.choice(fn, nm, replace=False) if nm > 0 else np.array([], dtype=int)
    return ci, mi


# ── Plot helpers ─────────────────────────────────────────────────────────────
def plot_shap_timeseries(shap_vals, X_samples, true_labels, pred_labels,
                         cls_name, title, save_path, max_plots=5):
    """Overlay SHAP attributions (for target class output) on ECG signal."""
    from matplotlib.patches import Patch
    n = min(len(X_samples), max_plots)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight='bold')
    for i in range(n):
        ax  = axes[i, 0]
        sig = X_samples[i].squeeze()
        sv  = shap_vals[i].squeeze()
        t   = np.arange(len(sig))
        
        # Plot ECG signal
        ax.plot(t, sig, color='#1565C0', linewidth=1.5, label='ECG', zorder=3)
        
        # Calculate limits and scale SHAP values
        ymin, ymax = float(sig.min() - 0.1), float(sig.max() + 0.1)
        ax.set_ylim(ymin, ymax)
        
        max_sv = np.max(np.abs(sv)) if np.max(np.abs(sv)) > 1e-9 else 1.0
        norm_sv = sv / max_sv
        
        # Draw opacity-scaled axvspan segments
        for j in range(len(t) - 1):
            val = float((norm_sv[j] + norm_sv[j+1]) / 2.0)
            alpha_val = float(min(0.4, abs(val) * 0.4))
            color = '#D32F2F' if val > 0 else '#388E3C'
            ax.axvspan(t[j], t[j+1], ymin=0.0, ymax=1.0, alpha=alpha_val, color=color, linewidth=0)
            
        tl = CLASS_NAMES.get(true_labels[i], str(true_labels[i]))
        pl = CLASS_NAMES.get(pred_labels[i],  str(pred_labels[i]))
        ax.set_title(f"Sample {i+1}  True: {tl}  Pred: {pl}", fontsize=9)
        ax.set_ylabel("Amp", fontsize=8)
        ax.grid(alpha=0.15)
        
        # Show legend on every subplot for clarity, or just the first one
        legend_elements = [
            plt.Line2D([0], [0], color='#1565C0', lw=1.5, label='ECG'),
            Patch(facecolor='#D32F2F', alpha=0.3, label='+SHAP (Supports Class)'),
            Patch(facecolor='#388E3C', alpha=0.3, label='-SHAP (Opposes Class)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=7, ncol=3)
        
    axes[-1, 0].set_xlabel("Time Step", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {save_path.name}")


def plot_shap_summary(mean_importance, T, out_dir):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(np.arange(T), mean_importance, color='#7B1FA2', alpha=0.8, width=0.9)
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Mean |SHAP|", fontsize=12)
    ax.set_title("SHAP Feature Importance — All Classes (Mean |SHAP| per Timestep)",
                 fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_plot.png", dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  [OK] shap_summary_plot.png")


# Helper to normalize SHAP output shape
def normalize_shap_output(sv_raw, n_classes):
    """Normalize SHAP output to a list of length n_classes, each of shape (n_samples, T, 1)."""
    if isinstance(sv_raw, list):
        return sv_raw
    if isinstance(sv_raw, np.ndarray):
        # SHAP versions differ: can return (n_classes, n, T, 1) or (n, T, 1, n_classes)
        if sv_raw.shape[0] == n_classes:
            return [sv_raw[c] for c in range(n_classes)]
        elif sv_raw.shape[-1] == n_classes:
            return [sv_raw[..., c] for c in range(n_classes)]
    return sv_raw


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    config_path = Path("configs/experiment_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_id = get_run_id(cfg)
    paths  = build_paths(run_id)
    paths["out_explain"].mkdir(parents=True, exist_ok=True)

    exp_cfg      = cfg["explainability"]
    n_background = exp_cfg.get("shap_background_samples", 200)
    n_correct    = exp_cfg.get("shap_correct_per_class",   10)
    n_mis        = exp_cfg.get("shap_misclassified_per_class", 5)
    shap_classes = exp_cfg.get("shap_classes", [0, 1, 2, 3])

    # CPU budget: cap background and summary to keep runtime < 10 min
    n_background = min(n_background, 100)   # GradientExplainer is heavier per-sample
    n_summary    = 200                       # samples for global importance
    # Number of background resamplings to average SHAP values.
    # GradientExplainer uses sampled baselines → attribution varies run-to-run.
    # Averaging over multiple resampled backgrounds stabilises Jaccard(SHAP, IG).
    n_shap_runs  = exp_cfg.get("shap_averaging_runs", 3)

    seed = cfg.get("seed", 42)
    rng  = np.random.default_rng(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu")   # GradientExplainer requires grad on inputs → CPU safer
    print(f"Device: {device} | Run ID: {run_id}")

    # Load model
    print("Loading model...")
    model, _ = load_hmr_bilstm(RLSTM_CKPT, device)
    wrapper   = ModelWrapper(model).to(device)
    wrapper.eval()

    # Load test data (using centralized INTER_TEST path)
    print("Loading test data...")
    print(f"  Using: {INTER_TEST}")
    test   = np.load(INTER_TEST)
    X_test = test["X"].astype(np.float32)
    y_test = test["y"].astype(np.int64)
    T_len  = X_test.shape[1]

    # Load training data for background (using centralized INTER_TRAIN path)
    print("Loading training data for background...")
    print(f"  Using: {INTER_TRAIN}")
    train = np.load(INTER_TRAIN)
    X_train = train["X"].astype(np.float32)

    # Get all predictions
    print("Getting predictions...")
    all_preds = []
    X_t = torch.from_numpy(X_test)
    with torch.no_grad():
        for i in range(0, len(X_t), 256):
            b = X_t[i:i+256].to(device)
            all_preds.append(wrapper(b).argmax(dim=-1).cpu().numpy())
    preds_all = np.concatenate(all_preds)
    print(f"  Accuracy: {(preds_all == y_test).mean():.4f}")

    # ── Select samples for detailed per-class analysis ──
    print("Selecting samples for detailed per-class analysis...")
    sample_indices = {}
    all_selected_idx = []
    for cls in shap_classes:
        ci, mi = select_samples(X_test, y_test, preds_all, cls,
                                n_correct=n_correct, n_mis=n_mis, rng=rng)
        sample_indices[cls] = {"correct": ci, "misclassified": mi}
        all_selected_idx.extend(ci)
        all_selected_idx.extend(mi)

    unique_selected_idx = np.array(sorted(list(set(all_selected_idx))))
    n_details = len(unique_selected_idx)
    print(f"  Selected {n_details} unique samples for detailed visualization.")

    # We will compute SHAP on X_sum and X_details together to save time and ensure consistency
    X_sum = torch.from_numpy(X_test[rng.choice(len(X_test), n_summary, replace=False)]).to(device)
    X_details = torch.from_numpy(X_test[unique_selected_idx]).to(device)
    X_all_explain = torch.cat([X_sum, X_details], dim=0)
    n_all = len(X_all_explain)

    # GradientExplainer: average over multiple background resamplings.
    # Each resampling uses a deterministic seed offset to ensure reproducibility.
    # This stabilises the top-K timestep set used in Jaccard(SHAP, IG).
    print(f"Computing SHAP on {n_all} samples averaged over {n_shap_runs} background resamplings...")
    
    n_classes = 5  # model has 5 classes
    shap_runs = []  # list of shap_vals per run, each a list/ndarray of length n_classes
    for run_i in range(n_shap_runs):
        run_seed = seed + run_i * 1000
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        bg_idx = np.random.choice(len(X_train), n_background, replace=False)
        X_bg   = torch.from_numpy(X_train[bg_idx]).to(device)
        
        explainer = shap.GradientExplainer(wrapper, X_bg)
        sv_raw = explainer.shap_values(X_all_explain)
        sv = normalize_shap_output(sv_raw, n_classes)
        shap_runs.append(sv)
        print(f"  Run {run_i+1}/{n_shap_runs} done (seed={run_seed}, bg_idx[0]={bg_idx[0]})")

    # Average across runs: shap_vals_all[c] shape (n_all, T, 1)
    shap_vals_all = [
        np.mean([shap_runs[r][c] for r in range(n_shap_runs)], axis=0)
        for c in range(n_classes)
    ]
    print(f"  SHAP averaging complete. Stability note: averaged over {n_shap_runs} runs.")

    # Split back to sum and details
    shap_vals_summary = [sv[:n_summary] for sv in shap_vals_all]
    shap_vals_details = [sv[n_summary:] for sv in shap_vals_all]

    # Global importance: mean |SHAP| over classes 0-3 and samples
    mean_imp = np.stack(
        [np.abs(shap_vals_summary[c]).squeeze(-1).mean(axis=0) for c in shap_classes], axis=0
    ).mean(axis=0)  # (T,)

    plot_shap_summary(mean_imp, T_len, paths["out_explain"])

    # Top-K CSV
    top_k   = 20
    top_idx = np.argsort(mean_imp)[::-1][:top_k]
    with open(paths["out_explain"] / "shap_importance_ranking.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "timestep", "mean_abs_shap"])
        for rank, idx in enumerate(top_idx, 1):
            w.writerow([rank, int(idx), f"{mean_imp[idx]:.6f}"])
    print(f"  [OK] shap_importance_ranking.csv")

    # Per-class plots
    top_per_class = {}
    for cls in shap_classes:
        cn = CLASS_NAMES[cls]
        print(f"Per-class SHAP: {cn}...")
        ci = sample_indices[cls]["correct"]
        mi = sample_indices[cls]["misclassified"]

        if len(ci) > 0:
            # Map test indices to positions in unique_selected_idx
            pos_ci = [np.where(unique_selected_idx == idx)[0][0] for idx in ci]
            svc = shap_vals_details[cls][pos_ci]   # (n, T, 1)
            plot_shap_timeseries(
                svc, X_test[ci], y_test[ci], preds_all[ci], cn,
                f"SHAP — Correct Predictions  Class {cn}",
                paths["out_explain"] / f"shap_class_{cn}.png",
                max_plots=min(5, n_correct)
            )
        else:
            print(f"  No correct samples for {cn}")

        if len(mi) > 0:
            # Map test indices to positions in unique_selected_idx
            pos_mi = [np.where(unique_selected_idx == idx)[0][0] for idx in mi]
            svm = shap_vals_details[cls][pos_mi]   # (n, T, 1)
            plot_shap_timeseries(
                svm, X_test[mi], y_test[mi], preds_all[mi], cn,
                f"SHAP — Misclassified  Class {cn}  (Why does the model fail?)",
                paths["out_explain"] / f"shap_misclassified_{cn}.png",
                max_plots=min(5, n_mis)
            )
        else:
            print(f"  No misclassified samples for {cn}")

        cls_imp = np.abs(shap_vals_summary[cls]).squeeze(-1).mean(axis=0)
        top10    = np.argsort(cls_imp)[::-1][:10].tolist()
        top_per_class[cn] = {
            "top10_timesteps": top10,
            "mean_importance": float(cls_imp.mean()),
            "correct_found":   int(len(ci)),
            "misclassified_found": int(len(mi))
        }

    # results.json
    results_json = {
        "experiment_version": cfg["experiment"]["version"],
        "run_id": run_id,
        "checkpoint_hash": get_checkpoint_hash(RLSTM_CKPT),
        "module": "explainability",
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "shap_summary": "shap_summary_plot.png",
            "top_global_timesteps": top_idx[:3].tolist(),
            "per_class": top_per_class
        }
    }
    with open(paths["out_explain"] / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    print(f"  [OK] results.json")

    print("\nSHAP analysis completed.")


if __name__ == "__main__":
    main()
