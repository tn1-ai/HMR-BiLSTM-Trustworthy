"""
Huấn luyện HMR-BiLSTM trên MIT-BIH ECG.

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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)
from pathlib import Path

from hmr_bilstm import RLSTMClassifier, RLSTMLoss


# ─── Config ───────────────────────────────────────────────────────────────────
CONFIG = {
    "data_dir":                "data/processed",
    "checkpoint_dir":          "results/checkpoints",
    "log_dir":                 "results/logs",
    "seed":                    42,
    "batch_size":              128,
    "hidden_size":             96,
    "dropout":                 0.25,
    "learning_rate":           1e-3,
    "min_lr":                  1e-5,
    "weight_decay":            1e-4,
    "lambda_smooth":           0.003,

    "epochs":                  45,
    "early_stopping_patience": 8,
    "grad_clip":               1.0,
    "num_classes":             5,
    "input_size":              1,
    "use_class_weights":       True,
    "cnn_out_channels":        64,
    "num_layers":              2,
    "use_focal_loss":          True,
    "focal_gamma":             1.5,
    # Adversarial training
    "adversarial_training":    True,
    "adv_epsilon":             0.02,
    "adv_ratio":               0.3,
}


# ─── Dataset ──────────────────────────────────────────────────────────────────
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
    # Deterministic flags — quan trọng cho research reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cosine_lr(epoch, total_epochs, base_lr, min_lr):
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
        "accuracy":        accuracy_score(y_true, preds),
        "precision_macro": precision_score(y_true, preds, average="macro", zero_division=0),
        "recall_macro":    recall_score(y_true, preds, average="macro", zero_division=0),
        "f1_macro":        f1_score(y_true, preds, average="macro", zero_division=0),
        "f1_weighted":     f1_score(y_true, preds, average="weighted", zero_division=0),
    }
    try:
        metrics["auc_ovr"] = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except ValueError:
        metrics["auc_ovr"] = 0.0
    return metrics


# ─── FGSM Adversarial Training ────────────────────────────────────────────────
def fgsm_attack_train(model, x, y, epsilon, criterion):
    """
    Generate FGSM adversarial examples during training.

    Uses the actual training criterion (Focal + class_weights) to compute
    gradients, ensuring adversarial perturbations align with the loss
    landscape the model is optimizing. Model stays in train mode to
    preserve BatchNorm running statistics.
    """
    x_adv = x.clone().detach().requires_grad_(True)

    # Forward pass để lấy gradient — model vẫn ở train mode
    # torch.enable_grad() đảm bảo gradient tính được ngay cả khi
    # hàm này được gọi từ trong torch.no_grad() context
    with torch.enable_grad():
        logits = model(x_adv)
        # r_fwd=None: only task loss, no smoothness penalty
        loss, _ = criterion(logits, y, r_fwd=None, r_bwd=None)
        
        # Clear model gradients before backward pass for perturbation
        # để gradient không tích lũy vào model parameters
        model.zero_grad()
        loss.backward()

    perturbation = epsilon * x_adv.grad.sign()

    return (x + perturbation).detach()


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip,
                    adv_training=False, adv_epsilon=0.02, adv_ratio=0.3):
    model.train()
    total_loss, total_task, total_smooth, n_samples = 0.0, 0.0, 0.0, 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        # --- Adversarial Training: mix clean + adversarial ---
        if adv_training and adv_epsilon > 0:
            split = int(len(X) * (1 - adv_ratio))
            X_clean, y_clean = X[:split],    y[:split]
            X_adv_src, y_adv = X[split:],    y[split:]

            if len(X_adv_src) > 0:

                X_adv = fgsm_attack_train(
                    model, X_adv_src, y_adv, adv_epsilon, criterion
                )
                X = torch.cat([X_clean, X_adv], dim=0)
                y = torch.cat([y_clean, y_adv], dim=0)

        optimizer.zero_grad()

        logits, internals = model(X, return_internals=True)
        loss, comp = criterion(
            logits, y,
            r_fwd=internals["r_fwd"],
            r_bwd=internals["r_bwd"],
        )

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        bs = X.size(0)
        total_loss   += loss.item() * bs
        total_task   += comp["task"].item() * bs
        total_smooth += comp["smooth"].item() * bs
        n_samples    += bs

    return {
        "loss":   total_loss  / max(1, n_samples),
        "task":   total_task  / max(1, n_samples),
        "smooth": total_smooth / max(1, n_samples),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    cfg = CONFIG
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Path(cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["log_dir"]).mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print("\n[Loading data]")
    train_ds = ECGDataset(f"{cfg['data_dir']}/splits/inter_train.npz")
    val_ds   = ECGDataset(f"{cfg['data_dir']}/splits/inter_val.npz")
    test_ds  = ECGDataset(f"{cfg['data_dir']}/splits/inter_test.npz")

    input_size = train_ds.X.shape[-1]
    seq_len    = train_ds.X.shape[1]
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"  Input dim: {input_size}, Sequence length: {seq_len}")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=0, pin_memory=False)

    # ── Class weights ──
    class_weights = None
    if cfg["use_class_weights"]:
        cw_path = Path(cfg["data_dir"]) / "class_weights.npy"
        if cw_path.exists():
            class_weights = torch.from_numpy(np.load(cw_path)).float().to(device)
            print(f"  Class weights: {class_weights.cpu().numpy()}")

    # ── Model ──
    print("\n[Building HMR-BiLSTM]")
    model = RLSTMClassifier(
        input_size=input_size,
        hidden_size=cfg["hidden_size"],
        dropout=cfg["dropout"],
        num_classes=cfg["num_classes"],
        cnn_out_channels=cfg["cnn_out_channels"],
        num_layers=cfg["num_layers"],
    ).to(device)
    print(f"  Parameters: {count_params(model):,}")

    # ── Loss ──

    criterion = RLSTMLoss(
        lambda_smooth=cfg["lambda_smooth"],
        class_weights=class_weights,
        use_focal=cfg["use_focal_loss"],
        focal_gamma=cfg["focal_gamma"],
    )
    loss_type = "Focal" if cfg["use_focal_loss"] else "CE"
    print(f"  Loss: {loss_type} (gamma={cfg['focal_gamma']})")
    if cfg["adversarial_training"]:
        print(f"  Adv-Training: FGSM eps={cfg['adv_epsilon']} ratio={cfg['adv_ratio']}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    # ── Training loop ──
    print(f"\n[Training] {cfg['epochs']} epochs, patience={cfg['early_stopping_patience']}")
    best_f1, best_epoch, patience_counter = 0.0, 0, 0
    history = []
    checkpoint_path = Path(cfg["checkpoint_dir"]) / "inter_best_rlstm.pt"

    for epoch in range(1, cfg["epochs"] + 1):
        current_lr = cosine_lr(epoch - 1, cfg["epochs"],
                               cfg["learning_rate"], cfg["min_lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        t0 = time.time()
        train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, device, cfg["grad_clip"],
            adv_training=cfg["adversarial_training"],
            adv_epsilon=cfg["adv_epsilon"],
            adv_ratio=cfg["adv_ratio"],
        )
        val_metrics = evaluate(model, val_loader, device, cfg["num_classes"])
        elapsed = time.time() - t0

        history.append({
            "epoch":        epoch,
            "lr":           current_lr,
            "train_loss":   train_stats["loss"],
            "train_task":   train_stats["task"],
            "train_smooth": train_stats["smooth"],
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "time_sec":     elapsed,
        })

        marker = ""
        if val_metrics["f1_macro"] > best_f1:
            best_f1, best_epoch, patience_counter = val_metrics["f1_macro"], epoch, 0
            torch.save({
                "model_state": model.state_dict(),
                "config":      cfg,
                "epoch":       epoch,
                "val_f1_macro": best_f1,
            }, checkpoint_path)
            marker = " <-- best"
        else:
            patience_counter += 1

        print(f"Epoch {epoch:3d} | lr={current_lr:.5f} | loss={train_stats['loss']:.4f} | "
              f"val: acc={val_metrics['accuracy']:.4f} "
              f"f1_mac={val_metrics['f1_macro']:.4f} "
              f"auc={val_metrics['auc_ovr']:.4f} | {elapsed:.0f}s{marker}")

        if patience_counter >= cfg["early_stopping_patience"]:
            print(f"\n[Early stop] epoch {epoch} (best={best_epoch}, F1={best_f1:.4f})")
            break

    # ── Test evaluation ──
    print(f"\n[Loading best checkpoint — epoch {best_epoch}]")
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)

    print("\n[Test set evaluation]")
    test_metrics = evaluate(model, test_loader, device, cfg["num_classes"])
    for k, v in test_metrics.items():
        print(f"  {k:<20}: {v:.4f}")

    # Per-class report
    model.eval()
    all_preds, all_y = [], []
    with torch.no_grad():
        for X, y in test_loader:
            all_preds.append(model(X.to(device)).argmax(-1).cpu().numpy())
            all_y.append(y.numpy())
    report = classification_report(
        np.concatenate(all_y), np.concatenate(all_preds),
        target_names=["N", "S", "V", "F", "Q"],
        zero_division=0, digits=4,
    )
    print("\n[Per-class report]\n" + report)

    # ── Save logs ──
    log_path = Path(cfg["log_dir"]) / "training_history.json"
    with open(log_path, "w") as f:
        json.dump({
            "config":              cfg,
            "history":             history,
            "best_epoch":          best_epoch,
            "best_val_f1_macro":   best_f1,
            "test_metrics":        test_metrics,
            "classification_report": report,
        }, f, indent=2)
    print(f"\n[OK] Logs -> {log_path}")
    print(f"[OK] Checkpoint -> {checkpoint_path}")


if __name__ == "__main__":
    main()