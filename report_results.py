"""
Tổng hợp kết quả và xuất hình ảnh cho MIT-BIH ECG (multi-class).

Cách chạy (sau khi đã chạy train.py và run_baselines.py):
    python report_results.py

Sinh các file:
- results/figures/confusion_matrix.png  (5x5 cho 5 classes)
- results/figures/roc_curve.png         (5 đường ROC one-vs-rest)
- results/figures/gate_trajectories.png (per-class residual gate)
- results/figures/comparison_bars.png   (so sánh các mô hình)
"""

import json
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize

from rlstm_model import RLSTMClassifier


CLASS_NAMES = ["N", "S", "V", "F", "Q"]
NUM_CLASSES = 5


# ─── Load mô hình HMR-BiLSTM đã train ────────────────────────
def load_rlstm_model(checkpoint_path, device, input_size=1):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = RLSTMClassifier(
        input_size=cfg.get("input_size", input_size),
        hidden_size=cfg["hidden_size"],
        dropout=cfg["dropout"],
        num_classes=cfg["num_classes"],
    ).to(device)
    try:
        model.load_state_dict(ckpt["model_state"], strict=False)
    except RuntimeError as e:
        print(f"[WARNING] Checkpoint incompatible: {e}")
        print("[WARNING] Using untrained model for inspection only.")
    model.eval()
    return model, ckpt


@torch.no_grad()
def collect_predictions_and_gates(model, X, y, device, batch_size=128):
    """Chạy mô hình trên test set và thu thập predictions + r_t."""
    X_tensor = torch.from_numpy(X).float()

    all_probs, all_preds, all_r = [], [], []
    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i:i+batch_size].to(device)
        logits, internals = model(batch, return_internals=True)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()
        r_fwd = internals["r_fwd"].cpu().numpy()
        r_bwd = internals["r_bwd"].cpu().numpy()
        r_combined = ((r_fwd + r_bwd) / 2).mean(axis=-1)
        all_probs.append(probs)
        all_preds.append(preds)
        all_r.append(r_combined)

    return np.concatenate(all_preds), np.concatenate(all_probs), np.concatenate(all_r)


# ─── Plotting functions ───────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    # Normalize theo hàng để xem tỉ lệ per-class
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Counts
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=10)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("True Class", fontsize=11)
    ax.set_title("Confusion Matrix (counts)", fontsize=12)

    # Normalized
    ax = axes[1]
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color=color, fontsize=10)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted Class", fontsize=11)
    ax.set_ylabel("True Class", fontsize=11)
    ax.set_title("Confusion Matrix (normalized)", fontsize=12)

    plt.suptitle("HMR-BiLSTM on MIT-BIH ECG Test Set", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved: {save_path}")


def plot_roc_curve(y_true, y_prob, save_path):
    """Multi-class ROC (one-vs-rest)."""
    y_bin = label_binarize(y_true, classes=range(NUM_CLASSES))

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#1F3864", "#C00000", "#2E75B6", "#E97132", "#43A047"]

    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                label=f"{CLASS_NAMES[i]} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves (One-vs-Rest) — HMR-BiLSTM on MIT-BIH ECG",
                 fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved: {save_path}")


def plot_gate_trajectories(r_all, y_all, save_path):
    """Vẽ trung bình ± std của r_t cho từng class."""
    fig, ax = plt.subplots(figsize=(10, 5))
    T = r_all.shape[1]
    t_axis = np.arange(T)

    colors = ["#1F3864", "#C00000", "#2E75B6", "#E97132", "#43A047"]

    for cls in range(NUM_CLASSES):
        mask = y_all == cls
        if mask.sum() == 0:
            continue
        # Subsample nếu quá nhiều để tránh overcrowding
        if mask.sum() > 2000:
            idx = np.where(mask)[0]
            sampled = np.random.choice(idx, 2000, replace=False)
            data = r_all[sampled]
        else:
            data = r_all[mask]
        mean = data.mean(axis=0)
        ax.plot(t_axis, mean, color=colors[cls],
                label=f"{CLASS_NAMES[cls]} (n={mask.sum()})",
                linewidth=2, alpha=0.8)

    ax.set_xlabel("Time Step (t)", fontsize=12)
    ax.set_ylabel(r"Residual Gate Mean $\bar{r}_t$", fontsize=12)
    ax.set_title("Residual Gate Trajectories Per Class (MIT-BIH ECG)", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved: {save_path}")


def plot_comparison_bars(all_results, save_path):
    """Bar chart so sánh tất cả mô hình."""
    model_order = ["logistic_regression", "decision_tree", "lstm", "bilstm", "rlstm"]
    model_labels = ["LR", "DT", "LSTM", "BiLSTM", "HMR-BiLSTM"]
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "auc_ovr"]
    metric_labels = ["Accuracy", "Prec(macro)", "Rec(macro)", "F1(macro)", "AUC(OvR)"]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(metrics))
    width = 0.15

    colors = ["#9E9E9E", "#FFA726", "#42A5F5", "#5C6BC0", "#43A047"]
    for i, (m, label) in enumerate(zip(model_order, model_labels)):
        if m not in all_results:
            continue
        values = [all_results[m][met] for met in metrics]
        bars = ax.bar(x + i * width, values, width, label=label, color=colors[i])
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=0)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison on MIT-BIH ECG Test Set", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved: {save_path}")


def plot_results_table(all_results, save_path):
    """Xuất bảng kết quả ra file ảnh."""
    model_order = ["logistic_regression", "decision_tree", "lstm", "bilstm", "rlstm"]
    pretty = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "lstm": "LSTM",
        "bilstm": "BiLSTM",
        "rlstm": "HMR-BiLSTM",
    }
    
    cell_text = []
    row_labels = []
    
    for name in model_order:
        if name not in all_results:
            continue
        m = all_results[name]
        marker = " *" if name == "rlstm" else ""
        row_labels.append(pretty[name] + marker)
        cell_text.append([
            f"{m['accuracy']:.4f}",
            f"{m['precision_macro']:.4f}",
            f"{m['recall_macro']:.4f}",
            f"{m['f1_macro']:.4f}",
            f"{m['f1_weighted']:.4f}",
            f"{m['auc_ovr']:.4f}"
        ])
        
    columns = ["Accuracy", "Precision", "Recall", "F1 (macro)", "F1 (weighted)", "AUC"]
    
    fig, ax = plt.subplots(figsize=(10, len(row_labels) * 0.5 + 1.5))
    ax.axis("tight")
    ax.axis("off")
    
    table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)
    
    plt.title("FINAL RESULTS — MIT-BIH ECG Test Set", fontsize=14, pad=20, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved: {save_path}")



# ─── Main ─────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("[Loading HMR-BiLSTM and test data]")
    try:
        test = np.load("data/processed/test.npz")
        X_test, y_test = test["X"], test["y"]
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return

    input_size = X_test.shape[-1] if len(X_test.shape) > 2 else 1
    checkpoint_path = "results/checkpoints/best_rlstm.pt"
    model, ckpt = load_rlstm_model(checkpoint_path, device, input_size)

    print(f"  Best epoch: {ckpt['epoch']}, val F1_macro: {ckpt.get('val_f1_macro', 0):.4f}")

    preds, probs, r_traj = collect_predictions_and_gates(
        model, X_test, y_test, device
    )

    print("\n[Generating figures]")
    plot_confusion_matrix(y_test, preds, fig_dir / "confusion_matrix.png")
    plot_roc_curve(y_test, probs, fig_dir / "roc_curve.png")
    plot_gate_trajectories(r_traj, y_test, fig_dir / "gate_trajectories.png")

    print("\n[Aggregating all results]")
    all_results = {}
    baseline_file = Path("results/baseline_results.json")
    if baseline_file.exists():
        with open(baseline_file) as f:
            all_results = json.load(f)

    all_results["rlstm"] = {
        "accuracy":         accuracy_score(y_test, preds),
        "precision_macro":  precision_score(y_test, preds, average="macro", zero_division=0),
        "recall_macro":     recall_score(y_test, preds, average="macro", zero_division=0),
        "f1_macro":         f1_score(y_test, preds, average="macro", zero_division=0),
        "f1_weighted":      f1_score(y_test, preds, average="weighted", zero_division=0),
        "auc_ovr":          roc_auc_score(y_test, probs, multi_class="ovr", average="macro"),
    }

    plot_comparison_bars(all_results, fig_dir / "comparison_bars.png")
    plot_results_table(all_results, fig_dir / "final_results_table.png")

    print("\n" + "=" * 80)
    print(" FINAL RESULTS — MIT-BIH ECG Test Set ")
    print("=" * 80)
    print(f"{'Model':<22} {'Acc':>8} {'P_mac':>8} {'R_mac':>8} {'F1_mac':>8} {'F1_w':>8} {'AUC':>8}")
    print("-" * 80)
    order = ["logistic_regression", "decision_tree", "lstm", "bilstm", "rlstm"]
    pretty = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "lstm": "LSTM",
        "bilstm": "BiLSTM",
        "rlstm": "HMR-BiLSTM",
    }
    for name in order:
        if name not in all_results:
            continue
        m = all_results[name]
        marker = " *" if name == "rlstm" else ""
        print(f"{pretty[name]:<22} "
              f"{m['accuracy']:>8.4f} "
              f"{m['precision_macro']:>8.4f} "
              f"{m['recall_macro']:>8.4f} "
              f"{m['f1_macro']:>8.4f} "
              f"{m['f1_weighted']:>8.4f} "
              f"{m['auc_ovr']:>8.4f}{marker}")

    csv_path = Path("results/final_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Accuracy", "Precision_macro", "Recall_macro",
                         "F1_macro", "F1_weighted", "AUC_OvR"])
        for name in order:
            if name not in all_results:
                continue
            m = all_results[name]
            writer.writerow([
                pretty[name],
                f"{m['accuracy']:.4f}",
                f"{m['precision_macro']:.4f}",
                f"{m['recall_macro']:.4f}",
                f"{m['f1_macro']:.4f}",
                f"{m['f1_weighted']:.4f}",
                f"{m['auc_ovr']:.4f}",
            ])
    print(f"\n[OK] CSV results saved to {csv_path}")
    print(f"[OK] All figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
