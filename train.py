"""
Huấn luyện R-LSTM trên MIT-BIH ECG (PHIÊN BẢN TỐI ƯU TỐC ĐỘ).

Cấu hình cân bằng:
- hidden_size = 96 (đủ capacity, không quá lớn)
- batch_size = 128 (nhanh nhưng vẫn ổn định)
- learning_rate = 1e-3 + cosine schedule (hội tụ nhanh)
- epochs = 20 với patience = 5 (đủ để hội tụ)
- Gradient clipping mạnh hơn (0.5) để chống explosion

Cách chạy:
    python train.py
"""

import os
import time
import json
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)
from pathlib import Path

from rlstm_model import RLSTMClassifier, RLSTMLoss


# ─── Cấu hình TỐI ƯU ───

CONFIG = {
    "data_dir": "data/processed",
    "checkpoint_dir": "results/checkpoints",
    "log_dir": "results/logs",
    "seed": 42,
    "batch_size": 128,
    "hidden_size": 96,
    "dropout": 0.25,
    "learning_rate": 1e-3,
    "min_lr": 1e-5,
    "weight_decay": 1e-4,
    "lambda_smooth": 0.003,        # Giảm vì T giảm 4x (187→46), L_smooth nhỏ hơn
    "epochs": 25,                  # Tăng vì model phức tạp hơn
    "early_stopping_patience": 6,
    "grad_clip": 1.0,              # Có thể nới lỏng vì CNN+RNN ổn định hơn
    "num_classes": 5,
    "use_class_weights": True,
    "cnn_out_channels": 64,        # MỚI: số kênh CNN output
}


# ─── Dataset ───
class ECGDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = torch.from_numpy(data["X"]).float()
        self.y = torch.from_numpy(data["y"]).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cosine_lr(epoch, total_epochs, base_lr, min_lr):
    """Cosine learning rate schedule."""
    progress = epoch / max(1, total_epochs)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device, num_classes=5):
    model.eval()
    all_logits, all_y = [], []
    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        all_logits.append(logits.cpu())
        all_y.append(y)

    logits = torch.cat(all_logits)
    y_true = torch.cat(all_y).numpy()
    probs  = torch.softmax(logits, dim=-1).numpy()
    preds  = logits.argmax(dim=-1).numpy()

    metrics = {
        "accuracy":         accuracy_score(y_true, preds),
        "precision_macro":  precision_score(y_true, preds, average="macro", zero_division=0),
        "recall_macro":     recall_score(y_true, preds, average="macro", zero_division=0),
        "f1_macro":         f1_score(y_true, preds, average="macro", zero_division=0),
        "f1_weighted":      f1_score(y_true, preds, average="weighted", zero_division=0),
    }
    try:
        metrics["auc_ovr"] = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except ValueError:
        metrics["auc_ovr"] = 0.0
    return metrics


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss, total_task, total_smooth, n_samples = 0.0, 0.0, 0.0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        logits, internals = model(X, return_internals=True)
        loss, comp = criterion(
            logits, y,
            r_fwd=internals["r_fwd"],
            r_bwd=internals["r_bwd"],
        )

        # Safety check: nếu loss bị NaN thì skip batch này
        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        bs = X.size(0)
        total_loss   += loss.item() * bs
        total_task   += comp["task"].item() * bs
        total_smooth += comp["smooth"].item() * bs
        n_samples += bs

    return {
        "loss":   total_loss / max(1, n_samples),
        "task":   total_task / max(1, n_samples),
        "smooth": total_smooth / max(1, n_samples),
    }


def main():
    cfg = CONFIG
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Path(cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["log_dir"]).mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[Loading data]")
    train_ds = ECGDataset(f"{cfg['data_dir']}/train.npz")
    val_ds   = ECGDataset(f"{cfg['data_dir']}/val.npz")
    test_ds  = ECGDataset(f"{cfg['data_dir']}/test.npz")

    input_size = train_ds.X.shape[-1]
    seq_len = train_ds.X.shape[1]
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"  Input dim: {input_size}, Sequence length: {seq_len}")

    # CRITICAL: num_workers=0 + pin_memory=False để tránh treo trên Windows
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=0, pin_memory=False)

    # Class weights (đã được clip trong preprocess)
    class_weights = None
    if cfg["use_class_weights"]:
        cw_path = Path(cfg["data_dir"]) / "class_weights.npy"
        if cw_path.exists():
            class_weights = torch.from_numpy(np.load(cw_path)).float().to(device)
            print(f"  Class weights (clipped): {class_weights.cpu().numpy()}")

    # Build model
    print("\n[Building R-LSTM model]")
    model = RLSTMClassifier(
    input_size=input_size,
    hidden_size=cfg["hidden_size"],
    dropout=cfg["dropout"],
    num_classes=cfg["num_classes"],
    cnn_out_channels=cfg["cnn_out_channels"],   # MỚI
    ).to(device)
    print(f"  Parameters: {count_params(model):,}")

    criterion = RLSTMLoss(
        lambda_smooth=cfg["lambda_smooth"],
        class_weights=class_weights,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    # Training loop
    print(f"\n[Training R-LSTM] {cfg['epochs']} epochs, patience={cfg['early_stopping_patience']}")
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []
    checkpoint_path = Path(cfg["checkpoint_dir"]) / "best_rlstm.pt"

    for epoch in range(1, cfg["epochs"] + 1):
        # Cosine LR schedule
        current_lr = cosine_lr(epoch - 1, cfg["epochs"],
                               cfg["learning_rate"], cfg["min_lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        t0 = time.time()
        train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, device, cfg["grad_clip"]
        )
        val_metrics = evaluate(model, val_loader, device, cfg["num_classes"])
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_stats["loss"],
            "train_task": train_stats["task"],
            "train_smooth": train_stats["smooth"],
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "time_sec": elapsed,
        })

        marker = ""
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "val_f1_macro": best_f1,
            }, checkpoint_path)
            marker = " <- best"
        else:
            patience_counter += 1

        print(f"Epoch {epoch:3d} | lr={current_lr:.5f} | loss={train_stats['loss']:.4f} | "
              f"val: acc={val_metrics['accuracy']:.4f} "
              f"f1_mac={val_metrics['f1_macro']:.4f} "
              f"auc={val_metrics['auc_ovr']:.4f} | {elapsed:.0f}s{marker}")

        if patience_counter >= cfg["early_stopping_patience"]:
            print(f"\n[Early stop] at epoch {epoch} (best at {best_epoch}, F1={best_f1:.4f})")
            break

    # Final test evaluation
    print(f"\n[Loading best checkpoint from epoch {best_epoch}]")
    if checkpoint_path.exists():
        try:
            ckpt = torch.load(checkpoint_path, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
        except RuntimeError as exc:
            print("[WARNING] Không thể load checkpoint vì kiến trúc model đã thay đổi:", exc)
            print("[WARNING] Sẽ dùng mô hình hiện tại để đánh giá, nhưng checkpoint cũ sẽ bị bỏ qua.")
            try:
                checkpoint_path.unlink()
            except Exception:
                pass
    else:
        print("[WARNING] Không tìm thấy checkpoint sau huấn luyện. Đánh giá bằng mô hình hiện tại.")

    print("\n[Evaluating on test set]")
    test_metrics = evaluate(model, test_loader, device, cfg["num_classes"])
    print(f"  Accuracy:       {test_metrics['accuracy']:.4f}")
    print(f"  Precision_macro: {test_metrics['precision_macro']:.4f}")
    print(f"  Recall_macro:    {test_metrics['recall_macro']:.4f}")
    print(f"  F1_macro:        {test_metrics['f1_macro']:.4f}")
    print(f"  F1_weighted:     {test_metrics['f1_weighted']:.4f}")
    print(f"  AUC_OvR:         {test_metrics['auc_ovr']:.4f}")

    # Per-class report
    print("\n[Per-class classification report]")
    model.eval()
    all_preds, all_y = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            preds = model(X).argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_y.append(y.numpy())
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_preds)
    target_names = ["N", "S", "V", "F", "Q"]
    report = classification_report(y_true, y_pred, target_names=target_names,
                                    zero_division=0, digits=4)
    print(report)

    # Save logs
    log_path = Path(cfg["log_dir"]) / "training_history.json"
    with open(log_path, "w") as f:
        json.dump({
            "config": cfg,
            "history": history,
            "best_epoch": best_epoch,
            "best_val_f1_macro": best_f1,
            "test_metrics": test_metrics,
            "classification_report": report,
        }, f, indent=2)
    print(f"\n[OK] Saved logs and checkpoint.")


if __name__ == "__main__":
    main()
