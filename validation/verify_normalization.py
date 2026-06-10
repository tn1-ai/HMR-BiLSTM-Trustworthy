# validation/verify_normalization.py
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def verify():
    raw_train_path = "data/raw/mitbih_train.csv"
    raw_test_path = "data/raw/mitbih_test.csv"
    processed_dir = Path("data/processed")
    out_dir = Path("outputs/normalization")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("--- Verifying Train-Only Normalization ---")
    
    # 1. Load raw datasets
    print("Loading raw CSVs to calculate unnormalized statistics...")
    df_train = pd.read_csv(raw_train_path, header=None)
    df_test = pd.read_csv(raw_test_path, header=None)
    
    X_train_full = df_train.iloc[:, :-1].values.astype(np.float32)
    y_train_full = df_train.iloc[:, -1].values.astype(np.int64)
    X_test_raw = df_test.iloc[:, :-1].values.astype(np.float32)
    
    # Replicate train/val split
    # Note: preprocess.py might subsample if SUBSAMPLE_SIZE is set, but by default it uses full.
    # We will replicate the split without subsampling (since SUBSAMPLE_SIZE is None by default).
    VAL_RATIO = 0.15
    RANDOM_SEED = 42
    
    X_train_raw, X_val_raw, _, _ = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VAL_RATIO,
        stratify=y_train_full,
        random_state=RANDOM_SEED,
    )
    
    # Calculate unnormalized statistics
    raw_stats = {
        "train": {
            "mean": float(X_train_raw.mean()),
            "std": float(X_train_raw.std())
        },
        "val": {
            "mean": float(X_val_raw.mean()),
            "std": float(X_val_raw.std())
        },
        "test": {
            "mean": float(X_test_raw.mean()),
            "std": float(X_test_raw.std())
        }
    }
    
    print("\nRaw Stats (Before Normalization):")
    print(f"  Train: mean={raw_stats['train']['mean']:.6f}, std={raw_stats['train']['std']:.6f}")
    print(f"  Val:   mean={raw_stats['val']['mean']:.6f}, std={raw_stats['val']['std']:.6f}")
    print(f"  Test:  mean={raw_stats['test']['mean']:.6f}, std={raw_stats['test']['std']:.6f}")
    
    # 2. Load processed datasets
    print("\nLoading processed datasets...")
    train_proc = np.load(processed_dir / "train.npz")
    val_proc = np.load(processed_dir / "val.npz")
    test_proc = np.load(processed_dir / "test.npz")
    
    X_train_proc = train_proc["X"]
    X_val_proc = val_proc["X"]
    X_test_proc = test_proc["X"]
    
    # Calculate processed statistics
    proc_stats = {
        "train": {
            "mean": float(X_train_proc.mean()),
            "std": float(X_train_proc.std())
        },
        "val": {
            "mean": float(X_val_proc.mean()),
            "std": float(X_val_proc.std())
        },
        "test": {
            "mean": float(X_test_proc.mean()),
            "std": float(X_test_proc.std())
        }
    }
    
    print("\nProcessed Stats (After Normalization):")
    print(f"  Train: mean={proc_stats['train']['mean']:.6f}, std={proc_stats['train']['std']:.6f}")
    print(f"  Val:   mean={proc_stats['val']['mean']:.6f}, std={proc_stats['val']['std']:.6f}")
    print(f"  Test:  mean={proc_stats['test']['mean']:.6f}, std={proc_stats['test']['std']:.6f}")
    
    # Load saved norm statistics
    norm_mean = float(np.load(processed_dir / "norm_mean.npy")[0])
    norm_std = float(np.load(processed_dir / "norm_std.npy")[0])
    
    print(f"\nSaved Normalization Parameters:")
    print(f"  norm_mean: {norm_mean:.6f}")
    print(f"  norm_std:  {norm_std:.6f}")
    
    # 3. Verification checks
    # Train mean should be ~0, std should be ~1
    train_mean_ok = abs(proc_stats["train"]["mean"]) < 5e-4
    train_std_ok = abs(proc_stats["train"]["std"] - 1.0) < 5e-4
    
    # Val and Test should be normalized using train mean and std
    expected_val_mean = (raw_stats["val"]["mean"] - norm_mean) / norm_std
    val_check_ok = abs(proc_stats["val"]["mean"] - expected_val_mean) < 5e-4
    
    leakage_detected = False
    # If the original preprocessing had leakage, val_proc.mean() would be exactly 0 (or closer to 0 than expected_val_mean)
    # because it normalized using the combined train+val mean.
    
    report = {
        "status": "SUCCESS" if (train_mean_ok and train_std_ok and val_check_ok) else "FAILED",
        "saved_parameters": {
            "mean": norm_mean,
            "std": norm_std
        },
        "raw_stats": raw_stats,
        "processed_stats": proc_stats,
        "verification": {
            "train_mean_is_zero": bool(train_mean_ok),
            "train_std_is_one": bool(train_std_ok),
            "val_normalized_using_train_stats": bool(val_check_ok),
            "data_leakage_prevented": bool(train_mean_ok and val_check_ok)
        }
    }
    
    with open(out_dir / "normalization_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"\nNormalization verification: {report['status']}")
    print(f"Report saved to {out_dir / 'normalization_report.json'}")

if __name__ == "__main__":
    verify()
