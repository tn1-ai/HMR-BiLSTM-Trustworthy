import os
import json
import yaml
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from configs.paths import get_run_id, build_paths, RLSTM_CKPT, get_checkpoint_hash, INTER_VAL
from evaluate_fgsm import build_test_loader
from report_results import load_hmr_bilstm
from calibration.calibration_metrics import compute_all_metrics
from calibration.reliability_diagram import plot_reliability_before_after, plot_per_class_reliability

class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, model, val_loader, device, lr=0.01, max_iter=50):
        """Fits temperature scaling parameter on the validation set."""
        model.eval()
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                logits_list.append(logits.cpu())
                labels_list.append(y)
                
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        nll_criterion = nn.CrossEntropyLoss()
        
        def eval_val():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss
            
        optimizer.step(eval_val)
        
        # Clamp temperature to avoid extremely small or negative values
        with torch.no_grad():
            self.temperature.clamp_(min=1e-3)
            
        print(f"Optimal temperature: {self.temperature.item():.4f}")
        return self.temperature.item()

def build_val_loader(batch_size=128):
    val = np.load(INTER_VAL)
    X_val, y_val = val["X"], val["y"]
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

def get_predictions_and_logits(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_labels.append(y)
            
    return torch.cat(all_logits), torch.cat(all_labels)

def main():
    # Load configuration
    config_path = Path("configs/experiment_config.yaml")
    if not config_path.exists():
        print("Error: configs/experiment_config.yaml not found.")
        return
        
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    run_id = get_run_id(cfg)
    paths = build_paths(run_id)
    paths["out_calib"].mkdir(parents=True, exist_ok=True)
    paths["out_figures"].mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model and loaders
    print("Loading HMR-BiLSTM model checkpoint...")
    model, _ = load_hmr_bilstm(RLSTM_CKPT, device)
    
    print("Preparing validation and test loaders...")
    val_loader = build_val_loader(batch_size=128)
    test_loader = build_test_loader(batch_size=128)
    
    # 1. Fit Temperature Scaling on validation set
    print("Fitting Temperature Scaling on validation set...")
    lr = cfg["calibration"].get("temp_lr", 0.01)
    max_iter = cfg["calibration"].get("temp_max_iter", 50)
    num_bins = cfg["calibration"].get("num_bins", 15)
    
    ts_model = TemperatureScaling().to(device)
    optimal_temp = ts_model.fit(model, val_loader, device, lr=lr, max_iter=max_iter)
    
    # Save temperature to JSON
    temp_json_path = paths["out_calib"] / "temperature.json"
    with open(temp_json_path, "w", encoding="utf-8") as f:
        json.dump({"temperature": optimal_temp}, f, indent=2)
    print(f"Saved temperature to {temp_json_path}")
    
    # 2. Evaluate on test set before and after scaling
    print("Evaluating calibration on test set...")
    test_logits, test_labels = get_predictions_and_logits(model, test_loader, device)
    
    # Before scaling probs
    probs_before = F.softmax(test_logits, dim=-1).numpy()
    labels_np = test_labels.numpy()
    
    # After scaling probs
    scaled_logits = test_logits / optimal_temp
    probs_after = F.softmax(scaled_logits, dim=-1).numpy()
    
    # Compute metrics
    metrics_before = compute_all_metrics(probs_before, labels_np, num_bins=num_bins)
    metrics_after = compute_all_metrics(probs_after, labels_np, num_bins=num_bins)
    
    print(f"\nCalibration metrics (Before vs After temperature scaling):")
    print(f"  ECE:   {metrics_before['ece']:.4f} -> {metrics_after['ece']:.4f}")
    print(f"  MCE:   {metrics_before['mce']:.4f} -> {metrics_after['mce']:.4f}")
    print(f"  NLL:   {metrics_before['nll']:.4f} -> {metrics_after['nll']:.4f}")
    print(f"  Brier: {metrics_before['brier']:.4f} -> {metrics_after['brier']:.4f}")
    
    print(f"\nClasswise ECE (After scaling):")
    for c, ece_c in metrics_after["classwise_ece"].items():
        print(f"  Class {c}: {ece_c:.4f}")
    
    # 3. Save results to JSON (matching JSON report contract)
    results_json = {
        "experiment_version": cfg["experiment"]["version"],
        "run_id": run_id,
        "checkpoint_hash": get_checkpoint_hash(RLSTM_CKPT),
        "module": "calibration",
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "ece_before": metrics_before["ece"],
            "ece_after": metrics_after["ece"],
            "mce_before": metrics_before["mce"],
            "mce_after": metrics_after["mce"],
            "nll_before": metrics_before["nll"],
            "nll_after": metrics_after["nll"],
            "brier_before": metrics_before["brier"],
            "brier_after": metrics_after["brier"],
            "classwise_ece_after": metrics_after["classwise_ece"],
            "temperature": optimal_temp
        }
    }
    
    results_json_path = paths["out_calib"] / "results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    print(f"Saved results contract to {results_json_path}")
    
    # 4. Generate reliability diagrams (saves PNGs and CSVs)
    print("Generating reliability diagrams...")
    plot_reliability_before_after(
        probs_before=probs_before,
        probs_after=probs_after,
        labels=labels_np,
        out_dir=paths["out_calib"],
        num_bins=num_bins,
        ece_before=metrics_before["ece"],
        ece_after=metrics_after["ece"],
        brier_before=metrics_before["brier"],
        brier_after=metrics_after["brier"]
    )
    
    print("Generating per-class reliability diagram...")
    plot_per_class_reliability(
        probs=probs_after,
        labels=labels_np,
        out_dir=paths["out_calib"],
        num_bins=num_bins,
        classwise_ece=metrics_after["classwise_ece"]
    )
    
    # Create copies in paths["out_figures"] for dashboard view convenience
    try:
        import shutil
        shutil.copy(paths["out_calib"] / "reliability_before_after.png", paths["out_figures"] / "reliability_before_after.png")
        shutil.copy(paths["out_calib"] / "reliability_before.png", paths["out_figures"] / "reliability_before.png")
        shutil.copy(paths["out_calib"] / "reliability_after.png", paths["out_figures"] / "reliability_after.png")
        shutil.copy(paths["out_calib"] / "reliability_per_class.png", paths["out_figures"] / "reliability_per_class.png")
        print("Copied reliability diagram figures to figures directory.")
    except Exception as e:
        print(f"Warning: could not copy figures to figures directory: {e}")
        
    print("\nCalibration module run completed successfully.")

if __name__ == "__main__":
    main()
