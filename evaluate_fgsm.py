from train import RLSTMLoss  # Import loss function từ train.py
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report
from torch.utils.data import DataLoader, TensorDataset

from report_results import load_hmr_bilstm
from run_baselines import LSTMBaseline
import torch.nn as nn
import os
from configs.paths import INTER_TEST


def fgsm_attack(model, x, y, epsilon, criterion):
    """Generate adversarial examples with FGSM using CORRECT loss function."""
    x_adv = x.clone().detach().requires_grad_(True)
    
    with torch.enable_grad():
        outputs = model(x_adv)
        loss, _ = criterion(outputs, y, r_fwd=None, r_bwd=None)  # Dùng criterion được truyền vào
        model.zero_grad()
        loss.backward()

    perturbation = epsilon * x_adv.grad.sign()
    x_adv = (x + perturbation).detach()
    delta = x_adv - x
    return x_adv, delta


# Clinically important classes (MIT-BIH label encoding):
# 0=N (Normal), 1=S (Supraventricular), 2=V (Ventricular), 3=F (Fusion), 4=Q (Unknown)
CLASS_NAMES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
# Classes whose per-class recall under attack is reported in the paper
CLINICAL_CLASSES = [1, 2, 3]  # S, V, F


def evaluate_fgsm(model, dataloader, device, criterion, epsilon=0.02):
    """Evaluate a model under FGSM attack at one epsilon.

    Returns aggregate metrics (accuracy, macro-F1, macro-recall, ASR,
    confidence) plus per-class recall for the clinically important classes
    S (1), V (2), F (3) — both clean and under attack.
    """
    model.eval()

    all_orig_preds = []
    all_preds = []
    all_labels = []
    orig_correct = []
    adv_wrong = []
    orig_confidence = []
    adv_confidence = []

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            orig_logits = model(x)
            orig_preds = orig_logits.argmax(dim=1)
            orig_probs = F.softmax(orig_logits, dim=1)
            orig_confidence.append(orig_probs.max(dim=1).values.cpu().numpy())

        if epsilon == 0.0:
            x_adv = x.clone().detach()
            delta = torch.zeros_like(x)
        else:
            x_adv, delta = fgsm_attack(model, x, y, epsilon, criterion)

        with torch.no_grad():
            adv_logits = model(x_adv)
            adv_preds = adv_logits.argmax(dim=1)
            adv_probs = F.softmax(adv_logits, dim=1)
            adv_confidence.append(adv_probs.max(dim=1).values.cpu().numpy())

        all_orig_preds.extend(orig_preds.cpu().numpy())
        all_preds.extend(adv_preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        orig_correct.append((orig_preds == y).cpu().numpy())
        adv_wrong.append((adv_preds != y).cpu().numpy())
        # delta is exactly epsilon * sign(x.grad) for FGSM without clamp
        # so we do not record linf per-sample here to avoid misleading metrics.

    all_orig_preds = np.array(all_orig_preds)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    orig_correct = np.concatenate(orig_correct)
    adv_wrong = np.concatenate(adv_wrong)
    orig_confidence = np.concatenate(orig_confidence)
    adv_confidence = np.concatenate(adv_confidence)

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    if orig_correct.sum() > 0:
        attack_success = (orig_correct & adv_wrong).sum() / orig_correct.sum()
    else:
        attack_success = 0.0

    # Per-class recall (clean vs adversarial) for clinically important classes.
    # recall_score with labels=[cls] gives per-class recall using all samples
    # as denominator for that class — equivalent to TP/(TP+FN).
    per_class_recall_clean = {}
    per_class_recall_adv = {}
    for cls in CLINICAL_CLASSES:
        name = CLASS_NAMES[cls]
        mask = all_labels == cls
        if mask.sum() > 0:
            per_class_recall_clean[f"recall_clean_{name}"] = float(
                (all_orig_preds[mask] == cls).sum() / mask.sum()
            )
            per_class_recall_adv[f"recall_adv_{name}"] = float(
                (all_preds[mask] == cls).sum() / mask.sum()
            )
        else:
            per_class_recall_clean[f"recall_clean_{name}"] = 0.0
            per_class_recall_adv[f"recall_adv_{name}"] = 0.0

    result = {
        "epsilon": epsilon,
        "label_accuracy": float(acc),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_recall": float(macro_recall),
        "attack_success_rate": float(attack_success),
        "avg_confidence": float(adv_confidence.mean()),
        "orig_avg_confidence": float(orig_confidence.mean()),
        "confidence_drop": float((orig_confidence - adv_confidence).mean()),
    }
    result.update(per_class_recall_clean)
    result.update(per_class_recall_adv)
    return result


def evaluate_fgsm_grid(model, dataloader, device, criterion, epsilons):
    results = []
    for epsilon in epsilons:
        print(f"Evaluating FGSM epsilon={epsilon}")
        result = evaluate_fgsm(model, dataloader, device, criterion, epsilon)
        results.append(result)
    return results


def plot_fgsm(results, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    eps = [r["epsilon"] for r in results]
    accuracy = [r["accuracy"] for r in results]
    macro_f1 = [r["macro_f1"] for r in results]
    macro_recall = [r["macro_recall"] for r in results]
    attack_rate = [r["attack_success_rate"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps, accuracy, marker="o", label="Accuracy")
    ax.plot(eps, macro_f1, marker="s", label="Macro F1")
    ax.plot(eps, macro_recall, marker="^", label="Macro Recall")
    ax.set_xlabel("Epsilon", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("FGSM Robustness Evaluation", fontsize=13)
    ax.set_ylim([0, 1.0])
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    fig_path = output_dir / "fgsm_robustness.png"
    plt.savefig(fig_path, dpi=400, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps, attack_rate, marker="d", color="#E64A19")
    ax.set_xlabel("Epsilon", fontsize=12)
    ax.set_ylabel("Attack Success Rate", fontsize=12)
    ax.set_title("FGSM Attack Success Rate", fontsize=13)
    ax.set_ylim([0, 1.0])
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    success_path = output_dir / "fgsm_attack_success_rate.png"
    plt.savefig(success_path, dpi=400, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    confidence = [r["avg_confidence"] for r in results]
    ax.plot(eps, confidence, marker="o", color="#6A1B9A", label="Avg Confidence")
    ax.set_xlabel("Epsilon", fontsize=12)
    ax.set_ylabel("Softmax Confidence", fontsize=12)
    ax.set_title("FGSM Confidence vs Epsilon", fontsize=13)
    ax.set_ylim([0, 1.0])
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    confidence_path = output_dir / "fgsm_confidence.png"
    plt.savefig(confidence_path, dpi=400, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    confidence_drop = [r["confidence_drop"] for r in results]
    ax.plot(eps, confidence_drop, marker="v", color="#283593", label="Confidence Drop")
    ax.set_xlabel("Epsilon", fontsize=12)
    ax.set_ylabel("Confidence Drop", fontsize=12)
    ax.set_title("FGSM Confidence Drop vs Epsilon", fontsize=13)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    confidence_drop_path = output_dir / "fgsm_confidence_drop.png"
    plt.savefig(confidence_drop_path, dpi=400, bbox_inches="tight")
    plt.close()

    print(f"Saved FGSM figure: {fig_path}")
    print(f"Saved FGSM attack success figure: {success_path}")
    print(f"Saved FGSM confidence figure: {confidence_path}")
    print(f"Saved FGSM confidence drop figure: {confidence_drop_path}")
    return fig_path, success_path, confidence_path, confidence_drop_path


def plot_fgsm_combined(results, output_dir: Path):
    """Create a consolidated 2x2 figure for FGSM metrics and save it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    eps = [r["epsilon"] for r in results]
    accuracy = [r["accuracy"] for r in results]
    macro_f1 = [r["macro_f1"] for r in results]
    macro_recall = [r["macro_recall"] for r in results]
    attack_rate = [r["attack_success_rate"] for r in results]
    confidence = [r.get("avg_confidence", 0.0) for r in results]
    confidence_drop = [r.get("confidence_drop", 0.0) for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(eps, accuracy, marker="o", label="Accuracy")
    ax.plot(eps, macro_f1, marker="s", label="Macro F1")
    ax.plot(eps, macro_recall, marker="^", label="Macro Recall")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Score")
    ax.set_title("Robustness: Scores vs Epsilon")
    ax.set_ylim([0, 1.0])
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--")

    ax = axes[0, 1]
    ax.plot(eps, attack_rate, marker="d", color="#E64A19")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Attack Success Rate")
    ax.set_title("Attack Success Rate vs Epsilon")
    ax.set_ylim([0, 1.0])
    ax.grid(alpha=0.3, linestyle="--")

    ax = axes[1, 0]
    ax.plot(eps, confidence, marker="o", label="Avg Confidence", color="#6A1B9A")
    ax.plot(eps, confidence_drop, marker="v", label="Confidence Drop", color="#283593")
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Softmax Confidence")
    ax.set_title("Confidence vs Epsilon")
    ax.set_ylim([0, 1.0])
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--")

    ax = axes[1, 1]
    ax.text(0.5, 0.5, "L-inf not reported\n(FGSM perturbation = \u03b5 × sign(grad))",
            horizontalalignment='center', verticalalignment='center', fontsize=11)
    ax.set_axis_off()

    plt.tight_layout()
    fused_path = output_dir / "fgsm_fused.png"
    plt.savefig(fused_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Saved fused FGSM figure: {fused_path}")
    return fused_path


def load_baseline_model(checkpoint_path, device):
    """Load LSTM / BiLSTM baseline saved state dict into LSTMBaseline."""
    state = torch.load(checkpoint_path, map_location=device)
    bidirectional = "bilstm" in str(checkpoint_path).lower()
    # Infer input_size dynamically from CNN conv1 weight
    # cnn.0.weight shape: (out_channels, in_channels, kernel) → in_channels = input_size
    raw_state = state if not (isinstance(state, dict) and "model_state" in state) else state["model_state"]
    input_size = raw_state["cnn.0.weight"].shape[1] if "cnn.0.weight" in raw_state else 1
    model = LSTMBaseline(
        input_size=input_size,
        hidden_size=96,
        bidirectional=bidirectional,
        dropout=0.25,
        num_classes=5,
    ).to(device)
    # state may be a plain state_dict saved by torch.save(best_state, path)
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        # strip module. prefix if present
        new_state = {k.replace("module.", ""): v for k, v in state.items()}
        # Convert weights
        model.load_state_dict(new_state, strict=True)
    else:
        try:
            model.load_state_dict(state, strict=True)
        except Exception:
            # if checkpoint contains nested keys, try common patterns
            if isinstance(state, dict) and "model_state" in state:
                model.load_state_dict(state["model_state"], strict=True)
            else:
                raise
    model.eval()
    return model, state


def plot_comparison(comparison_results, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"LSTM": "#1f77b4", "BiLSTM": "#ff7f0e", "HMR-BiLSTM": "#2ca02c"}
    markers = {"LSTM": "o", "BiLSTM": "s", "HMR-BiLSTM": "^"}
    for model_name, results in comparison_results.items():
        epsilons = [r["epsilon"] for r in results]
        macro_f1 = [r["macro_f1"] for r in results]
        ax.plot(epsilons, macro_f1, marker=markers.get(model_name, "o"), label=model_name,
                color=colors.get(model_name, None), linewidth=2.5, markersize=8)
    ax.set_xlabel("Epsilon", fontsize=12)
    ax.set_ylabel("Macro F1", fontsize=12)
    ax.set_title("Model Robustness Comparison: Macro F1 vs FGSM Epsilon", fontsize=13, fontweight="bold")
    ax.set_ylim([0, 1.0])
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    fig_path = output_dir / "fgsm_comparison_macro_f1.png"
    plt.savefig(fig_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison figure: {fig_path}")


def generate_table(comparison_results, output_dir: Path):
    rows = []
    for model_name, results in comparison_results.items():
        # assume epsilons [0.0, 0.02, 0.05]
        clean_result = results[0]
        fgsm_result = results[1] if len(results) > 1 else results[-1]
        clean_f1 = clean_result["macro_f1"]
        fgsm_f1 = fgsm_result["macro_f1"]
        asr = fgsm_result.get("attack_success_rate", 0.0)
        rows.append({"Model": model_name, "Clean F1": f"{clean_f1:.4f}", "FGSM F1": f"{fgsm_f1:.4f}", "ASR": f"{asr:.4f}"})

    csv_path = Path("results/tables/fgsm_comparison_table.csv"); csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["Model", "Clean F1", "FGSM F1", "ASR"]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(row[h] for h in header) + "\n")
    print(f"Saved comparison table: {csv_path}")


def compare_models(models, epsilons, device, test_loader, output_dir: Path):
    # ============================================================
    # TẠO CRITERION GIỐNG HỆT LÚC TRAIN
    # ============================================================
    from train import RLSTMLoss  # Đảm bảo import ở đầu file
    
    cw_path = Path("data/processed/class_weights.npy")
    if cw_path.exists():
        class_weights = torch.from_numpy(np.load(cw_path)).float().to(device)
        print(f"Loaded class weights: {class_weights.cpu().numpy()}")
    else:
        class_weights = None
        print("No class weights found, using uniform weights")
    
    criterion = RLSTMLoss(
        lambda_smooth=0.003,      # Giống CONFIG trong train.py
        class_weights=class_weights,
        use_focal=True,            # Dùng FocalLoss như lúc train
        focal_gamma=1.5,           # Giống train.py
    )
    print("Using FocalLoss (gamma=1.5) with class weights for FGSM attack")
    print("=" * 60)
    
    # ============================================================
    # ĐÁNH GIÁ CÁC MODEL
    # ============================================================
    comparison_results = {}
    for model_name, checkpoint_path in models.items():
        cp = Path(checkpoint_path)
        if not cp.exists():
            print(f"Warning: {checkpoint_path} not found, skipping {model_name}")
            continue
        print(f"\n[Evaluating {model_name}]")
        try:
            if model_name == "HMR-BiLSTM":
                model, _ = load_hmr_bilstm(checkpoint_path, device)
            else:
                model, _ = load_baseline_model(checkpoint_path, device)
            
            model_results = []
            for eps in epsilons:
                print(f"  epsilon={eps}")
                # Pass criterion to evaluate_fgsm
                res = evaluate_fgsm(model, test_loader, device, criterion, eps)
                model_results.append(res)
            comparison_results[model_name] = model_results
            print(f"✓ {model_name} completed")
        except Exception as e:
            print(f"✗ {model_name} failed: {e}")

    if not comparison_results:
        print("No models evaluated. Check checkpoint paths.")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = Path("results/logs/fgsm_comparison_results.json"); json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"\nSaved comparison results: {json_path}")
    plot_comparison(comparison_results, output_dir)
    generate_table(comparison_results, output_dir)


def save_results(results, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = Path("results/logs/fgsm_results.json"); json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path("results/tables/fgsm_results.csv"); csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    header = [
        "epsilon", "label_accuracy", "accuracy", "macro_f1", "macro_recall",
        "attack_success_rate", "avg_confidence", "orig_avg_confidence", "confidence_drop",
        "recall_clean_S", "recall_adv_S",
        "recall_clean_V", "recall_adv_V",
        "recall_clean_F", "recall_adv_F",
    ]
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in results:
            f.write(",".join(str(row.get(h, "")) for h in header) + "\n")

    print(f"Saved FGSM results JSON: {json_path}")
    print(f"Saved FGSM results CSV: {csv_path}")
    return json_path, csv_path


def plot_ecg_comparison(model, dataloader, device, epsilon, output_dir: Path, sample_index=0, criterion=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    x, y = next(iter(dataloader))
    x = x.to(device)
    y = y.to(device)
    if criterion is None:
        from train import RLSTMLoss
        cw_path = Path("data/processed/class_weights.npy")
        class_weights = torch.from_numpy(np.load(cw_path)).float().to(device) if cw_path.exists() else None
        criterion = RLSTMLoss(lambda_smooth=0.003, class_weights=class_weights, use_focal=True, focal_gamma=1.5)
    x_adv, _ = fgsm_attack(model, x, y, epsilon, criterion)

    x_orig = x[sample_index].cpu().squeeze(-1).numpy()
    x_adv = x_adv[sample_index].cpu().squeeze(-1).numpy()
    t = np.arange(x_orig.shape[-1]) if x_orig.ndim == 1 else np.arange(x_orig.shape[0])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, x_orig, label="Original ECG", linewidth=1.5)
    ax.plot(t, x_adv, label=f"Adversarial ECG (epsilon={epsilon})", linewidth=1.5)
    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel("Signal", fontsize=12)
    ax.set_title("Original vs Adversarial ECG", fontsize=13)
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    example_path = output_dir / f"fgsm_ecg_example_epsilon_{epsilon:.3f}.png"
    plt.savefig(example_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Saved ECG comparison figure: {example_path}")
    return example_path


def build_test_loader(batch_size=128):
    test = np.load(INTER_TEST)
    X_test, y_test = test["X"], test["y"]
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def main():
    # Fix seeds for reproducibility (FGSM gradient is deterministic, but
    # setting seeds guards against future PGD random restarts or shuffled loaders).
    torch.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser(description="Evaluate model robustness under FGSM attacks")
    parser.add_argument("--checkpoint", default="results/checkpoints/best_rlstm.pt",
                        help="Path to the trained model checkpoint")
    parser.add_argument("--epsilons", nargs="*", type=float,
                        default=[0.0, 0.01, 0.02, 0.05],
                        help="Epsilon values for FGSM attack")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for test evaluation")
    parser.add_argument("--output-dir", default="results/figures",
                        help="Directory to save FGSM figures and results")
    parser.add_argument("--compare", action="store_true", help="Run comparison across baseline models (LSTM, BiLSTM, HMR-BiLSTM)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("[Loading model]")
    model, _ = load_hmr_bilstm(args.checkpoint, device)

    print("[Preparing test loader]")
    test_loader = build_test_loader(batch_size=args.batch_size)

    if args.compare:
        models = {
            "LSTM": "results/checkpoints/best_lstm.pt",
            "BiLSTM": "results/checkpoints/best_bilstm.pt",
            "HMR-BiLSTM": "results/checkpoints/best_rlstm.pt",
        }
        compare_models(models, args.epsilons, device, test_loader, Path(args.output_dir))
        return

    print("[Starting FGSM evaluation]")
    # Build criterion (must match training config)
    from train import RLSTMLoss
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
    results = evaluate_fgsm_grid(model, test_loader, device, criterion, args.epsilons)

    print("\nFGSM evaluation results:")
    print("epsilon, label_accuracy, accuracy, macro_f1, macro_recall, attack_success_rate")
    for row in results:
        print(f"{row['epsilon']:.4f}, {row['label_accuracy']:.4f}, {row['accuracy']:.4f}, {row['macro_f1']:.4f}, {row['macro_recall']:.4f}, {row['attack_success_rate']:.4f}")

    output_dir = Path(args.output_dir)
    save_results(results, output_dir)
    # keep legacy individual plots but also create a fused figure
    fig_paths = plot_fgsm(results, output_dir)
    fused_path = plot_fgsm_combined(results, output_dir)

    # remove older individual FGSM plot files to reduce clutter
    to_remove = [
        output_dir / "fgsm_robustness.png",
        output_dir / "fgsm_attack_success_rate.png",
        output_dir / "fgsm_confidence.png",
        output_dir / "fgsm_confidence_drop.png",
    ]
    for p in to_remove:
        try:
            if p.exists():
                p.unlink()
                print(f"Removed old figure: {p}")
        except Exception:
            pass

    selected_epsilon = 0.02 if 0.02 in args.epsilons else args.epsilons[-1]
    plot_ecg_comparison(model, test_loader, device, selected_epsilon, output_dir, criterion=criterion)


if __name__ == "__main__":
    main()
