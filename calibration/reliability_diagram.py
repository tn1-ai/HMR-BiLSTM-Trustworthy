import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import csv

def compute_bin_stats(probs: np.ndarray, labels: np.ndarray, num_bins: int = 15):
    """Computes bin stats for reliability diagram."""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    bin_stats = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if bin_lower == 0.0:
            in_bin = in_bin | (confidences == 0.0)
            
        count = int(in_bin.sum())
        if count > 0:
            acc = float(accuracies[in_bin].mean())
            conf = float(confidences[in_bin].mean())
            bin_stats.append({
                "bin_center": float((bin_lower + bin_upper) / 2),
                "mean_confidence": conf,
                "mean_accuracy": acc,
                "sample_count": count
            })
        else:
            bin_stats.append({
                "bin_center": float((bin_lower + bin_upper) / 2),
                "mean_confidence": 0.0,
                "mean_accuracy": 0.0,
                "sample_count": 0
            })
            
    return bin_stats

def save_bin_stats_to_csv(bin_stats, csv_path: Path):
    """Saves bin statistics to a CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_center", "mean_confidence", "mean_accuracy", "sample_count"])
        for stat in bin_stats:
            writer.writerow([
                f"{stat['bin_center']:.4f}",
                f"{stat['mean_confidence']:.4f}",
                f"{stat['mean_accuracy']:.4f}",
                stat['sample_count']
            ])

def plot_reliability_before_after(probs_before: np.ndarray, probs_after: np.ndarray, 
                                  labels: np.ndarray, out_dir: Path, num_bins: int = 15,
                                  ece_before: float = 0.0, ece_after: float = 0.0,
                                  brier_before: float = 0.0, brier_after: float = 0.0):
    """Plots side-by-side reliability diagrams with confidence histograms at the bottom."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute stats
    stats_before = compute_bin_stats(probs_before, labels, num_bins)
    stats_after = compute_bin_stats(probs_after, labels, num_bins)
    
    # Save CSV files
    save_bin_stats_to_csv(stats_before, out_dir / "reliability_before.csv")
    save_bin_stats_to_csv(stats_after, out_dir / "reliability_after.csv")
    
    # Plot side-by-side
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1.2]}, sharex=True)
    fig.subplots_adjust(hspace=0.08, wspace=0.15)
    
    # Perfect calibration reference line
    axes[0, 0].plot([0, 1], [0, 1], 'k--', label="Perfectly Calibrated", linewidth=1.5)
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label="Perfectly Calibrated", linewidth=1.5)
    
    # Subplot 1: Before
    centers_b = [s['bin_center'] for s in stats_before if s['sample_count'] > 0]
    confs_b = [s['mean_confidence'] for s in stats_before if s['sample_count'] > 0]
    accs_b = [s['mean_accuracy'] for s in stats_before if s['sample_count'] > 0]
    counts_b = [s['sample_count'] for s in stats_before]
    centers_all = [s['bin_center'] for s in stats_before]
    
    axes[0, 0].plot(confs_b, accs_b, marker='o', color='#E64A19', linewidth=2.5, markersize=8, label="Model Calibration")
    # Draw gap bars
    for c, a in zip(confs_b, accs_b):
        axes[0, 0].plot([c, c], [c, a], color='red', alpha=0.3, linewidth=1.5)
        
    axes[0, 0].set_ylabel("Accuracy", fontsize=12)
    axes[0, 0].set_title(f"Before Temperature Scaling\n(ECE = {ece_before:.4f}, Brier = {brier_before:.4f})", fontsize=13, fontweight='bold')
    axes[0, 0].set_xlim([0, 1])
    axes[0, 0].set_ylim([0, 1.05])
    axes[0, 0].legend(loc="upper left")
    axes[0, 0].grid(alpha=0.3, linestyle="--")
    
    # Subplot 2: After
    centers_a = [s['bin_center'] for s in stats_after if s['sample_count'] > 0]
    confs_a = [s['mean_confidence'] for s in stats_after if s['sample_count'] > 0]
    accs_a = [s['mean_accuracy'] for s in stats_after if s['sample_count'] > 0]
    counts_a = [s['sample_count'] for s in stats_after]
    
    axes[0, 1].plot(confs_a, accs_a, marker='s', color='#2E7D32', linewidth=2.5, markersize=8, label="Model Calibration")
    # Draw gap bars
    for c, a in zip(confs_a, accs_a):
        axes[0, 1].plot([c, c], [c, a], color='green', alpha=0.3, linewidth=1.5)
        
    axes[0, 1].set_title(f"After Temperature Scaling\n(ECE = {ece_after:.4f}, Brier = {brier_after:.4f})", fontsize=13, fontweight='bold')
    axes[0, 1].set_xlim([0, 1])
    axes[0, 1].set_ylim([0, 1.05])
    axes[0, 1].legend(loc="upper left")
    axes[0, 1].grid(alpha=0.3, linestyle="--")
    
    # Frequency histograms at the bottom
    bin_width = 1.0 / num_bins
    axes[1, 0].bar(centers_all, counts_b, width=bin_width*0.8, color='#E64A19', alpha=0.75, edgecolor='black')
    axes[1, 0].set_xlabel("Mean Predicted Confidence", fontsize=12)
    axes[1, 0].set_ylabel("Sample Count", fontsize=11)
    axes[1, 0].grid(alpha=0.3, linestyle="--", axis='y')
    
    axes[1, 1].bar(centers_all, counts_a, width=bin_width*0.8, color='#2E7D32', alpha=0.75, edgecolor='black')
    axes[1, 1].set_xlabel("Mean Predicted Confidence", fontsize=12)
    axes[1, 1].grid(alpha=0.3, linestyle="--", axis='y')
    
    plt.suptitle("Reliability Diagram Before and After Calibration", fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(out_dir / "reliability_before_after.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save separate files for before/after as requested by the plan
    # Before PNG
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], 'k--', label="Perfectly Calibrated", linewidth=1.5)
    ax.plot(confs_b, accs_b, marker='o', color='#E64A19', linewidth=2.5, markersize=8, label="Model Calibration")
    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Reliability Diagram (Before)\nECE = {ece_before:.4f}, Brier = {brier_before:.4f}", fontsize=13, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, linestyle="--")
    plt.savefig(out_dir / "reliability_before.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # After PNG
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], 'k--', label="Perfectly Calibrated", linewidth=1.5)
    ax.plot(confs_a, accs_a, marker='s', color='#2E7D32', linewidth=2.5, markersize=8, label="Model Calibration")
    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Reliability Diagram (After)\nECE = {ece_after:.4f}, Brier = {brier_after:.4f}", fontsize=13, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, linestyle="--")
    plt.savefig(out_dir / "reliability_after.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved reliability plots and CSV to {out_dir}")

def plot_per_class_reliability(probs: np.ndarray, labels: np.ndarray, out_dir: Path, num_bins: int = 15, classwise_ece: dict = None):
    """Plots reliability diagrams for each class (one-vs-rest)."""
    num_classes = probs.shape[1]
    class_names = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
    
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
    
    for c in range(num_classes):
        ax = axes[c]
        
        # One-vs-rest binary problem
        binary_conf = probs[:, c]
        binary_labels = (labels == c).astype(int)
        
        # We can't reuse compute_bin_stats easily because it assumes max confidence
        # Here we use the specific class confidence
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_confs = []
        bin_accs = []
        
        for bl, bu in zip(bin_lowers, bin_uppers):
            in_bin = (binary_conf > bl) & (binary_conf <= bu)
            if bl == 0.0:
                in_bin = in_bin | (binary_conf == 0.0)
                
            if in_bin.sum() > 0:
                bin_confs.append(float(binary_conf[in_bin].mean()))
                bin_accs.append(float(binary_labels[in_bin].mean()))
                
        # Plot
        ax.plot([0, 1], [0, 1], 'k--', label="Perfect", linewidth=1.5)
        ax.plot(bin_confs, bin_accs, marker='o', color='#1976D2', linewidth=2.5)
        
        for conf, acc in zip(bin_confs, bin_accs):
            ax.plot([conf, conf], [conf, acc], color='blue', alpha=0.3, linewidth=1.5)
            
        ece_str = f"ECE: {classwise_ece[c]:.4f}" if classwise_ece else ""
        ax.set_title(f"Class {class_names.get(c, str(c))}\n{ece_str}", fontsize=13)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel("Confidence")
        if c == 0:
            ax.set_ylabel("Accuracy (Recall)")
        ax.grid(alpha=0.3, linestyle="--")
        
        # Highlight clinical classes S, V, F with red border
        if c in [1, 2, 3]:
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2)
                
    plt.suptitle("Per-Class Reliability Diagrams (One-vs-Rest)", fontsize=16, fontweight='bold', y=1.05)
    plt.savefig(out_dir / "reliability_per_class.png", dpi=300, bbox_inches='tight')
    plt.close()
