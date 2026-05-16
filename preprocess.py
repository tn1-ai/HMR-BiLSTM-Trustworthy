# preprocess.py (FIXED VERSION)

"""
Tiền xử lý MIT-BIH ECG cho HMR-BiLSTM (PHIÊN BẢN ĐÃ FIX).

Các cải tiến quan trọng:
1. ECG normalization (CRITICAL FIX)
2. Giữ statistics mean/std cho inference
3. Class weights clipping an toàn hơn
4. In thống kê normalization
5. Pipeline ổn định hơn cho HMR-BiLSTM

Cách chạy:
    python preprocess.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ─── Cấu hình ───
TRAIN_CSV = "data/raw/mitbih_train.csv"
TEST_CSV  = "data/raw/mitbih_test.csv"
OUT_DIR   = "data/processed"

VAL_RATIO   = 0.15
RANDOM_SEED = 42
SEQUENCE_LENGTH = 187
NUM_CLASSES = 5

# Subsample để train nhanh hơn
SUBSAMPLE_SIZE = 30000

CLASS_NAMES = {
    0: "N (Normal)",
    1: "S (Supraventricular)",
    2: "V (Ventricular)",
    3: "F (Fusion)",
    4: "Q (Unknown)",
}


def load_csv(path, name):
    print(f"  Loading {name}...")
    df = pd.read_csv(path, header=None)

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)

    print(f"    Shape: X={X.shape}, y={y.shape}")

    return X, y


def print_class_distribution(y, split_name):
    print(f"\n  Class distribution ({split_name}):")

    unique, counts = np.unique(y, return_counts=True)
    total = len(y)

    for cls, cnt in zip(unique, counts):
        pct = cnt / total * 100
        name = CLASS_NAMES.get(int(cls), f"Class {int(cls)}")

        print(f"    {name:<25} {cnt:>7} ({pct:.2f}%)")

    print(f"    {'Total':<25} {total:>7}")
    print(f"    Imbalance ratio: {counts.max() / counts.min():.1f}x")


def stratified_subsample(X, y, target_size, seed=42):
    """Lấy stratified subsample giữ tỷ lệ class."""

    if len(X) <= target_size:
        return X, y

    ratio = target_size / len(X)

    X_sub, _, y_sub, _ = train_test_split(
        X,
        y,
        train_size=ratio,
        stratify=y,
        random_state=seed,
    )

    return X_sub, y_sub


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" PREPROCESS MIT-BIH ECG (FIXED VERSION) ")
    print("=" * 70)

    # =========================================================
    # STEP 1: LOAD DATA
    # =========================================================

    print("\n[1/8] Loading CSV files...")

    X_train_full, y_train_full = load_csv(TRAIN_CSV, "train")
    X_test, y_test = load_csv(TEST_CSV, "test")

    print_class_distribution(y_train_full, "TRAIN (full)")
    print_class_distribution(y_test, "TEST")

    # =========================================================
    # STEP 2: SUBSAMPLE
    # =========================================================

    print(f"\n[2/8] Subsampling train to {SUBSAMPLE_SIZE} samples...")

    X_train_full, y_train_full = stratified_subsample(
        X_train_full,
        y_train_full,
        SUBSAMPLE_SIZE,
        RANDOM_SEED,
    )

    print_class_distribution(y_train_full, "TRAIN (after subsample)")

    # =========================================================
    # STEP 3: NORMALIZATION (CRITICAL FIX)
    # =========================================================

    print("\n[3/8] Normalizing ECG signals...")

    # IMPORTANT:
    # ECG signals should ALWAYS be normalized before RNN/LSTM.
    # Otherwise sigmoid/tanh gates saturate easily.

    mean = X_train_full.mean()
    std = X_train_full.std() + 1e-8

    print(f"  Mean BEFORE normalization: {mean:.6f}")
    print(f"  Std  BEFORE normalization: {std:.6f}")

    # Normalize using TRAIN statistics only
    X_train_full = (X_train_full - mean) / std
    X_test = (X_test - mean) / std

    print(f"  Mean AFTER normalization: {X_train_full.mean():.6f}")
    print(f"  Std  AFTER normalization: {X_train_full.std():.6f}")

    # Save normalization statistics
    np.save(
        out_dir / "norm_mean.npy",
        np.array([mean], dtype=np.float32),
    )

    np.save(
        out_dir / "norm_std.npy",
        np.array([std], dtype=np.float32),
    )

    # =========================================================
    # STEP 4: TRAIN/VAL SPLIT
    # =========================================================

    print("\n[4/8] Splitting train -> train + val...")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VAL_RATIO,
        stratify=y_train_full,
        random_state=RANDOM_SEED,
    )

    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")

    # =========================================================
    # STEP 5: RESHAPE FOR LSTM
    # =========================================================

    print("\n[5/8] Reshaping sequences for HMR-BiLSTM...")

    # LSTM input format:
    # (batch, timesteps, features)

    X_train = X_train.reshape(-1, SEQUENCE_LENGTH, 1).astype(np.float32)
    X_val   = X_val.reshape(-1, SEQUENCE_LENGTH, 1).astype(np.float32)
    X_test  = X_test.reshape(-1, SEQUENCE_LENGTH, 1).astype(np.float32)

    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")

    # =========================================================
    # STEP 6: CLASS WEIGHTS
    # =========================================================

    print("\n[6/8] Computing class weights...")

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=y_train,
    ).astype(np.float32)

    print("\n  Class weights BEFORE clipping:")

    for cls, w in enumerate(class_weights):
        print(
            f"    {CLASS_NAMES.get(cls, f'Class {cls}'):<25} weight = {w:.4f}"
        )

    # IMPORTANT:
    # Too aggressive clipping may hurt minority classes.

    class_weights = np.clip(class_weights, 0.5, 10.0)

    print("\n  Class weights AFTER clipping to [0.5, 10.0]:")

    for cls, w in enumerate(class_weights):
        print(
            f"    {CLASS_NAMES.get(cls, f'Class {cls}'):<25} weight = {w:.4f}"
        )

    # =========================================================
    # STEP 7: SAVE FILES
    # =========================================================

    print("\n[7/8] Saving processed files...")

    np.savez(out_dir / "train.npz", X=X_train, y=y_train)
    np.savez(out_dir / "val.npz",   X=X_val,   y=y_val)
    np.savez(out_dir / "test.npz",  X=X_test,  y=y_test)

    np.save(out_dir / "class_weights.npy", class_weights)

    # =========================================================
    # STEP 8: SUMMARY
    # =========================================================

    print("\n[8/8] Preprocessing complete.")

    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")

    print(f"  Feature dim: {X_train.shape[-1]}")
    print(f"  Number of classes: {NUM_CLASSES}")

    print("\n[IMPORTANT NEXT STEPS]")
    print("  1. Delete old processed files before retraining")
    print("  2. Set lambda_smooth = 0.0 for debugging")
    print("  3. Add .clone() in RLSTMLayer.forward")
    print("  4. Use orthogonal initialization for recurrent weights")


if __name__ == "__main__":
    main()
