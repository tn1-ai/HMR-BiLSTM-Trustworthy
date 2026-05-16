"""
Chạy các baselines cho MIT-BIH ECG (PHIÊN BẢN TỐI ƯU).

Cấu hình:
- LR + DT: chạy nhanh trên CPU (~1-2 phút mỗi cái)
- LSTM/BiLSTM: hidden=96, epochs=12, early stop patience=4
- num_workers=0 + pin_memory=False để tránh treo trên Windows

Cách chạy:
    python run_baselines.py
"""

import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)
from pathlib import Path


NUM_CLASSES = 5


def load_data():
    train = np.load("data/processed/train.npz")
    val   = np.load("data/processed/val.npz")
    test  = np.load("data/processed/test.npz")
    return (train["X"], train["y"]), (val["X"], val["y"]), (test["X"], test["y"])


def compute_metrics(y_true, y_pred, y_prob):
    metrics = {
        "accuracy":         accuracy_score(y_true, y_pred),
        "precision_macro":  precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro":     recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro":         f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted":      f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    try:
        metrics["auc_ovr"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        metrics["auc_ovr"] = 0.0
    return metrics


def flatten_sequences(X):
    return X.reshape(X.shape[0], -1)


# ─── Sklearn baselines ───
def run_logistic_regression(train, test):
    X_tr, y_tr = train
    X_te, y_te = test
    X_tr_flat = flatten_sequences(X_tr)
    X_te_flat = flatten_sequences(X_te)

    t0 = time.time()
    model = LogisticRegression(max_iter=5000, random_state=42,
                                class_weight="balanced", n_jobs=-1,
                                solver="lbfgs")
    model.fit(X_tr_flat, y_tr)
    train_time = time.time() - t0

    y_pred = model.predict(X_te_flat)
    y_prob = model.predict_proba(X_te_flat)
    metrics = compute_metrics(y_te, y_pred, y_prob)
    return {**metrics, "train_time_sec": train_time}


def run_decision_tree(train, test):
    X_tr, y_tr = train
    X_te, y_te = test
    X_tr_flat = flatten_sequences(X_tr)
    X_te_flat = flatten_sequences(X_te)

    t0 = time.time()
    model = DecisionTreeClassifier(max_depth=15, min_samples_leaf=10,
                                    class_weight="balanced", random_state=42)
    model.fit(X_tr_flat, y_tr)
    train_time = time.time() - t0

    y_pred = model.predict(X_te_flat)
    y_prob = model.predict_proba(X_te_flat)
    metrics = compute_metrics(y_te, y_pred, y_prob)
    return {**metrics, "train_time_sec": train_time}


# ─── LSTM baselines ───
class LSTMBaseline(nn.Module):
    """LSTM/BiLSTM baseline với CNN feature extractor (để fair so với HMR-BiLSTM)."""

    def __init__(self, input_size, hidden_size=96, bidirectional=False,
                 dropout=0.25, num_classes=5, cnn_out_channels=64):
        super().__init__()

        # CNN feature extractor (giống HMR-BiLSTM để fair)
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout * 0.5),
        )

        # LSTM hoặc BiLSTM trên features
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True,
        )

        out_dim = hidden_size * (2 if bidirectional else 1)

        # Attention pooling (giống HMR-BiLSTM để fair)
        self.attention = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.Tanh(),
            nn.Linear(out_dim // 2, 1),
        )

        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        # x: (B, T, C) → CNN → (B, T', C')
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)

        # LSTM
        h_seq, _ = self.lstm(x)  # (B, T', H)

        # Attention pooling
        scores = self.attention(h_seq)              # (B, T', 1)
        weights = torch.softmax(scores, dim=1)
        h_pooled = (h_seq * weights).sum(dim=1)     # (B, H)

        # Classification
        h_pooled = self.layer_norm(h_pooled)
        h_pooled = self.dropout(h_pooled)
        return self.classifier(h_pooled)


def train_lstm_baseline(name, train, val, test, bidirectional, device,
                         class_weights=None, epochs=12):
    X_tr, y_tr = train
    X_va, y_va = val
    X_te, y_te = test
    input_size = X_tr.shape[-1]

    train_ds = TensorDataset(torch.from_numpy(X_tr).float(),
                              torch.from_numpy(y_tr).long())
    val_ds   = TensorDataset(torch.from_numpy(X_va).float(),
                              torch.from_numpy(y_va).long())
    test_ds  = TensorDataset(torch.from_numpy(X_te).float(),
                              torch.from_numpy(y_te).long())

    # num_workers=0 để tránh treo trên Windows
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False,
                              num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False,
                              num_workers=0, pin_memory=False)

    torch.manual_seed(42)
    model = LSTMBaseline(
        input_size=input_size, hidden_size=96,
        bidirectional=bidirectional, dropout=0.25,
        num_classes=NUM_CLASSES,
    ).to(device)

    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_f1 = 0.0
    best_state = None
    patience = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                all_logits.append(model(X).cpu())
                all_y.append(y)
        logits = torch.cat(all_logits)
        y_true = torch.cat(all_y).numpy()
        preds = logits.argmax(-1).numpy()
        val_f1 = f1_score(y_true, preds, average="macro", zero_division=0)

        print(f"    epoch {epoch:2d} | val F1_macro = {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 4:  # Early stop patience
                break

    train_time = time.time() - t0
    print(f"  [{name}] best val F1_macro = {best_f1:.4f} (stopped at epoch {epoch})")

    # Test evaluation
    model.load_state_dict(best_state)
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            all_logits.append(model(X).cpu())
            all_y.append(y)
    logits = torch.cat(all_logits)
    y_true = torch.cat(all_y).numpy()
    probs = torch.softmax(logits, -1).numpy()
    preds = logits.argmax(-1).numpy()

    metrics = compute_metrics(y_true, preds, probs)
    return {**metrics, "train_time_sec": train_time}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n[Loading data]")
    train_data, val_data, test_data = load_data()
    print(f"  Train: {train_data[0].shape}, Test: {test_data[0].shape}")

    cw = torch.from_numpy(np.load("data/processed/class_weights.npy")).float()
    print(f"  Class weights: {cw.numpy()}")

    all_results = {}

    print("\n[Logistic Regression]")
    all_results["logistic_regression"] = run_logistic_regression(train_data, test_data)
    print(f"  Test F1_macro: {all_results['logistic_regression']['f1_macro']:.4f}, "
          f"AUC: {all_results['logistic_regression']['auc_ovr']:.4f}")

    print("\n[Decision Tree]")
    all_results["decision_tree"] = run_decision_tree(train_data, test_data)
    print(f"  Test F1_macro: {all_results['decision_tree']['f1_macro']:.4f}, "
          f"AUC: {all_results['decision_tree']['auc_ovr']:.4f}")

    print("\n[LSTM (unidirectional)]")
    all_results["lstm"] = train_lstm_baseline(
        "LSTM", train_data, val_data, test_data,
        bidirectional=False, device=device, class_weights=cw,
        epochs=12,
    )
    print(f"  Test F1_macro: {all_results['lstm']['f1_macro']:.4f}, "
          f"AUC: {all_results['lstm']['auc_ovr']:.4f}")

    print("\n[BiLSTM]")
    all_results["bilstm"] = train_lstm_baseline(
        "BiLSTM", train_data, val_data, test_data,
        bidirectional=True, device=device, class_weights=cw,
        epochs=12,
    )
    print(f"  Test F1_macro: {all_results['bilstm']['f1_macro']:.4f}, "
          f"AUC: {all_results['bilstm']['auc_ovr']:.4f}")

    Path("results").mkdir(exist_ok=True)
    out_path = Path("results/baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 78)
    print(" BASELINE COMPARISON SUMMARY - MIT-BIH ECG (macro avg) ")
    print("=" * 78)
    print(f"{'Model':<25} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'F1_w':>8} {'AUC':>8}")
    print("-" * 78)
    for name, m in all_results.items():
        print(f"{name:<25} "
              f"{m['accuracy']:>8.4f} "
              f"{m['precision_macro']:>8.4f} "
              f"{m['recall_macro']:>8.4f} "
              f"{m['f1_macro']:>8.4f} "
              f"{m['f1_weighted']:>8.4f} "
              f"{m['auc_ovr']:>8.4f}")
    print(f"\n[OK] Results saved to {out_path}")


if __name__ == "__main__":
    main()
