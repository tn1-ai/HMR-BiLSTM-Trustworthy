# validation/evaluate_splits.py
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

from hmr_bilstm import RLSTMClassifier, RLSTMLoss

# Hyperparameters for fast training on CPU
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_SUB_SIZE = 5000  # Subsample size for fast training on CPU

def get_dataloader(npz_path, shuffle=False, subsample_size=None):
    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]
    
    if subsample_size and len(X) > subsample_size:
        # Stratified subsampling
        np.random.seed(42)
        unique_classes, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        
        indices = []
        for c, prop in zip(unique_classes, proportions):
            c_indices = np.where(y == c)[0]
            n_sub = int(np.round(prop * subsample_size))
            n_sub = max(1, min(n_sub, len(c_indices)))
            sub_indices = np.random.choice(c_indices, n_sub, replace=False)
            indices.extend(sub_indices)
            
        indices = np.array(indices)
        X = X[indices]
        y = y[indices]
        
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)

def train_and_eval(split_name, train_path, val_path, test_path):
    print(f"\n--- Training HMR-BiLSTM on {split_name} Split ---")
    
    # Loader configuration
    train_loader = get_dataloader(train_path, shuffle=True, subsample_size=TRAIN_SUB_SIZE)
    val_loader = get_dataloader(val_path, shuffle=False)
    test_loader = get_dataloader(test_path, shuffle=False)
    
    # Model configuration
    model = RLSTMClassifier(
        input_size=1,
        hidden_size=96,
        num_layers=2,
        num_classes=5,
        dropout=0.25
    ).to(DEVICE)
    
    # RLSTMLoss with Focal Loss (no class weights for simplicity or same config)
    criterion = RLSTMLoss(
        lambda_smooth=0.003,
        class_weights=None,
        use_focal=True,
        focal_gamma=1.5
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Train loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            
            logits, internals = model(X, return_internals=True)
            loss, _ = criterion(logits, y, internals["r_fwd"], internals["r_bwd"])
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        epoch_time = time.time() - start_time
        
        # Fast validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(DEVICE)
                logits = model(X)
                val_preds.extend(logits.argmax(dim=-1).cpu().numpy())
                val_labels.extend(y.numpy())
                
        val_acc = accuracy_score(val_labels, val_preds)
        print(f"  Epoch {epoch}/{EPOCHS} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {epoch_time:.1f}s")
        
    # Final evaluation on Test Set
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(DEVICE)
            logits = model(X)
            test_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            test_labels.extend(y.numpy())
            
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    
    metrics = {
        "accuracy": float(accuracy_score(test_labels, test_preds)),
        "precision_macro": float(precision_score(test_labels, test_preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(test_labels, test_preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(test_labels, test_preds, average="macro", zero_division=0))
    }
    
    print(f"Evaluation complete for {split_name} split:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision_macro']:.4f}")
    print(f"  Recall:    {metrics['recall_macro']:.4f}")
    print(f"  Macro F1:  {metrics['f1_macro']:.4f}")
    
    return metrics

def main():
    splits_dir = Path("data/processed/splits")
    out_dir = Path("outputs/splits")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not (splits_dir / "intra_train.npz").exists():
        print("Processed splits not found. Please run preprocessing first.")
        return
        
    # Evaluate splits
    intra_metrics = train_and_eval(
        "Intra-patient (Beat split)",
        splits_dir / "intra_train.npz",
        splits_dir / "intra_val.npz",
        splits_dir / "intra_test.npz"
    )
    
    inter_metrics = train_and_eval(
        "Inter-patient (AAMI EC57)",
        splits_dir / "inter_train.npz",
        splits_dir / "inter_val.npz",
        splits_dir / "inter_test.npz"
    )
    
    # Save results
    with open(out_dir / "intra_patient_results.json", "w") as f:
        json.dump(intra_metrics, f, indent=2)
    with open(out_dir / "inter_patient_results.json", "w") as f:
        json.dump(inter_metrics, f, indent=2)
        
    # Print direct comparison table
    print("\n" + "="*50)
    print(" DIRECT PROTOCOL COMPARISON TABLE")
    print("="*50)
    print(f"| Metric          | Intra-Patient Split | Inter-Patient Split (AAMI) | Difference |")
    print(f"|-----------------|---------------------|----------------------------|------------|")
    for m in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        diff = inter_metrics[m] - intra_metrics[m]
        print(f"| {m:<15} | {intra_metrics[m]:>19.4f} | {inter_metrics[m]:>26.4f} | {diff:>10.4f} |")
    print("="*50)

if __name__ == "__main__":
    main()
