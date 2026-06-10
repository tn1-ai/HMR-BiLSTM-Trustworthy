# validation/pgd_convergence.py
import argparse
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import TensorDataset, DataLoader

from hmr_bilstm import RLSTMClassifier, RLSTMLoss
from report_results import load_hmr_bilstm
from evaluate_fgsm import build_test_loader
from configs.paths import INTER_TEST

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "results/checkpoints/inter_best_rlstm.pt"
NUM_TEST_SAMPLES = 2000  # Subset size to run quickly on CPU
STEPS_LIST = [10, 20, 50, 100]
EPSILON = 0.02
ALPHA = 0.005

def get_test_subset_loader(npz_path, subset_size=200):
    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]
    
    # Stratified subsampling
    np.random.seed(42)
    unique_classes, counts = np.unique(y, return_counts=True)
    proportions = counts / len(y)
    
    indices = []
    for c, prop in zip(unique_classes, proportions):
        c_indices = np.where(y == c)[0]
        n_sub = int(np.round(prop * subset_size))
        n_sub = max(1, min(n_sub, len(c_indices)))
        sub_indices = np.random.choice(c_indices, n_sub, replace=False)
        indices.extend(sub_indices)
        
    indices = np.array(indices)
    X_sub = torch.from_numpy(X[indices]).float()
    y_sub = torch.from_numpy(y[indices]).long()
    
    dataset = TensorDataset(X_sub, y_sub)
    return DataLoader(dataset, batch_size=64, shuffle=False)

def pgd_attack(model, x, y, epsilon, alpha, steps, criterion, data_min=None, data_max=None):
    x_adv = x.clone().detach()
    if data_min is None: data_min = x.min()
    if data_max is None: data_max = x.max()
    
    # Random initialization
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
            
    return x_adv

def run_study():
    out_dir = Path("outputs/robustness")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not Path(CHECKPOINT_PATH).exists():
        print(f"Error: Model checkpoint {CHECKPOINT_PATH} not found.")
        return
        
    print(f"Loading HMR-BiLSTM model from {CHECKPOINT_PATH}...")
    model, _ = load_hmr_bilstm(CHECKPOINT_PATH, DEVICE)
    model.eval()
    
    # Load class weights if available
    cw_path = Path("data/processed/class_weights.npy")
    if cw_path.exists():
        class_weights = torch.from_numpy(np.load(cw_path)).float().to(DEVICE)
    else:
        class_weights = None
        
    criterion = RLSTMLoss(
        lambda_smooth=0.003,
        class_weights=class_weights,
        use_focal=True,
        focal_gamma=1.5
    )
    
    test_loader = get_test_subset_loader(INTER_TEST, subset_size=NUM_TEST_SAMPLES)
    
    all_x = []
    for X, _ in test_loader:
        all_x.append(X)
    all_x = torch.cat(all_x)
    data_min = float(all_x.min())
    data_max = float(all_x.max())
    print(f"Global dataset min: {data_min:.4f}, max: {data_max:.4f} for PGD clamping")
    
    # First evaluate Clean (0 steps)
    print("Evaluating Clean performance...")
    clean_preds, clean_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(DEVICE)
            logits = model(X)
            clean_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            clean_labels.extend(y.numpy())
            
    clean_acc = accuracy_score(clean_labels, clean_preds)
    clean_f1 = f1_score(clean_labels, clean_preds, average="macro", zero_division=0)
    print(f"  Clean Accuracy: {clean_acc:.4f} | Clean Macro F1: {clean_f1:.4f}")
    
    results = {
        "epsilon": EPSILON,
        "alpha": ALPHA,
        "clean": {
            "accuracy": float(clean_acc),
            "f1_macro": float(clean_f1)
        },
        "convergence": []
    }
    
    # Evaluate PGD at different steps
    for steps in STEPS_LIST:
        print(f"Running PGD attack with {steps} steps...")
        start_time = time.time()
        adv_preds, adv_labels = [], []
        orig_correct_count = 0
        asr_numerator = 0
        
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            # Find clean predictions to compute ASR
            with torch.no_grad():
                clean_logits = model(X)
                clean_p = clean_logits.argmax(dim=-1)
                
            # Perform attack
            X_adv = pgd_attack(model, X, y, EPSILON, ALPHA, steps, criterion, data_min, data_max)
            
            with torch.no_grad():
                adv_logits = model(X_adv)
                adv_p = adv_logits.argmax(dim=-1)
                
            adv_preds.extend(adv_p.cpu().numpy())
            adv_labels.extend(y.cpu().numpy())
            
            # Attack Success Rate (ASR): proportion of correct clean predictions that become incorrect after attack
            correct_mask = (clean_p == y)
            orig_correct_count += correct_mask.sum().item()
            asr_numerator += ((clean_p == y) & (adv_p != y)).sum().item()
            
        elapsed = time.time() - start_time
        adv_acc = accuracy_score(adv_labels, adv_preds)
        adv_f1 = f1_score(adv_labels, adv_preds, average="macro", zero_division=0)
        asr = asr_numerator / orig_correct_count if orig_correct_count > 0 else 0.0
        
        print(f"  Steps: {steps:<3} | Acc: {adv_acc:.4f} | F1: {adv_f1:.4f} | ASR: {asr:.4f} | Time: {elapsed:.1f}s")
        
        results["convergence"].append({
            "steps": steps,
            "accuracy": float(adv_acc),
            "f1_macro": float(adv_f1),
            "attack_success_rate": float(asr),
            "time_sec": float(elapsed)
        })
        
    # Save results to JSON
    with open(out_dir / "pgd_convergence_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Generate convergence plot
    print("Generating convergence plot...")
    steps = [r["steps"] for r in results["convergence"]]
    accs = [r["accuracy"] for r in results["convergence"]]
    f1s = [r["f1_macro"] for r in results["convergence"]]
    asrs = [r["attack_success_rate"] for r in results["convergence"]]
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, accs, marker='o', color='blue', label='Accuracy', linewidth=2)
    plt.plot(steps, f1s, marker='s', color='green', label='Macro F1-score', linewidth=2)
    plt.plot(steps, asrs, marker='^', color='red', label='Attack Success Rate (ASR)', linewidth=2)
    
    plt.axhline(y=clean_acc, color='blue', linestyle='--', alpha=0.5, label='Clean Accuracy')
    plt.axhline(y=clean_f1, color='green', linestyle='--', alpha=0.5, label='Clean Macro F1')
    
    plt.title("PGD Attack Convergence Study (HMR-BiLSTM)", fontsize=14, fontweight='bold')
    plt.xlabel("PGD Attack Steps (Iterations)", fontsize=12)
    plt.ylabel("Performance Metric Value", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    
    plot_path = out_dir / "pgd_convergence_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Convergence plot saved to {plot_path}")

if __name__ == "__main__":
    run_study()
