"""
verify_checkpoint.py
====================
Chạy ngay sau khi train_inter_patient.py hoàn thành.

Mục đích:
  1. Xác nhận checkpoint load đúng key "model_state" (không chạy trên random weights)
  2. Đối chiếu accuracy/F1 với val F1 lúc train (ghi trong training_history.json)
  3. SHAP pilot với 5 sample từ inter_test.npz để xác nhận GradientExplainer pass qua RMC cell
  4. In sẵn class distribution để phát hiện sớm nếu model collapse vào class N

Usage:
    python verify_checkpoint.py
"""
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report

import shap

from report_results import load_hmr_bilstm
from configs.paths import RLSTM_CKPT, get_checkpoint_hash, INTER_VAL, INTER_TEST

CLASS_NAMES = ["N", "S", "V", "F", "Q"]
DEVICE = torch.device("cpu")


def load_training_history():
    hist_path = Path("results/logs/training_history.json")
    if not hist_path.exists():
        return None
    with open(hist_path, "r") as f:
        return json.load(f)


def run_inference(model, X, batch_size=256):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            b = torch.from_numpy(X[i:i+batch_size]).to(DEVICE)
            preds.append(model(b).argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def main():
    print("=" * 60)
    print("CHECKPOINT VERIFICATION")
    print("=" * 60)

    # ── 1. Load checkpoint ──
    ckpt_path = RLSTM_CKPT
    print(f"\n[1] Loading checkpoint: {ckpt_path}")

    if not Path(ckpt_path).exists():
        print(f"  ERROR: {ckpt_path} not found. Is training still running?")
        return

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    # Verify key structure
    required_keys = {"model_state", "epoch", "val_f1_macro", "config"}
    missing = required_keys - set(ckpt.keys())
    if missing:
        print(f"  ERROR: Checkpoint missing keys: {missing}")
        return

    saved_epoch  = ckpt["epoch"]
    saved_val_f1 = ckpt["val_f1_macro"]
    ckpt_hash    = get_checkpoint_hash(ckpt_path)
    print(f"  ✓ Keys OK: {list(ckpt.keys())}")
    print(f"  Saved epoch:   {saved_epoch}")
    print(f"  Saved val F1:  {saved_val_f1:.4f}")
    print(f"  Checkpoint ID: {ckpt_hash}")

    # ── 2. Load model ──
    print("\n[2] Loading model weights...")
    model, _ = load_hmr_bilstm(ckpt_path, DEVICE)
    model.eval()
    print("  ✓ Weights loaded successfully")

    # ── 3. Cross-check with training history ──
    print("\n[3] Cross-checking with training_history.json...")
    hist = load_training_history()
    if hist is None:
        print("  [SKIP] training_history.json not found")
    else:
        hist_best_f1 = hist.get("best_val_f1_macro", 0.0)
        hist_best_ep = hist.get("best_epoch", -1)
        if abs(hist_best_f1 - saved_val_f1) < 1e-4 and hist_best_ep == saved_epoch:
            print(f"  ✓ Checkpoint matches history (epoch {hist_best_ep}, F1={hist_best_f1:.4f})")
        else:
            print(f"  WARNING: Mismatch!")
            print(f"    Checkpoint → epoch={saved_epoch}, F1={saved_val_f1:.4f}")
            print(f"    History    → epoch={hist_best_ep}, F1={hist_best_f1:.4f}")

    # ── 4. Evaluate on inter_val.npz ──
    val_file = INTER_VAL
    print(f"\n[4] Evaluating on val set: {val_file}")
    val = np.load(val_file)
    X_val = val["X"].astype(np.float32)
    y_val = val["y"].astype(np.int64)

    val_preds = run_inference(model, X_val)
    val_acc   = accuracy_score(y_val, val_preds)
    val_f1    = f1_score(y_val, val_preds, average="macro", zero_division=0)
    print(f"  Val Accuracy:  {val_acc:.4f}")
    print(f"  Val Macro F1:  {val_f1:.4f}  (checkpoint saved: {saved_val_f1:.4f})")

    delta_f1 = abs(val_f1 - saved_val_f1)
    if delta_f1 > 0.02:
        print(f"  WARNING: ΔF1={delta_f1:.4f} > 0.02 — possible wrong checkpoint or data mismatch!")
    else:
        print(f"  ✓ ΔF1={delta_f1:.4f} within tolerance — weights verified")

    # ── 5. Per-class report on inter_test.npz ──
    test_file = INTER_TEST
    print(f"\n[5] Per-class report on test set: {test_file}")
    test = np.load(test_file)
    X_test = test["X"].astype(np.float32)
    y_test = test["y"].astype(np.int64)

    test_preds = run_inference(model, X_test)
    test_f1_macro = f1_score(y_test, test_preds, average="macro", zero_division=0)
    test_f1_pc    = f1_score(y_test, test_preds, average=None, zero_division=0)

    print(f"  Test Macro F1: {test_f1_macro:.4f}")
    for i, name in enumerate(CLASS_NAMES):
        f1_c = test_f1_pc[i] if i < len(test_f1_pc) else float("nan")
        flag = " ← clinically critical" if name in ("S", "V", "F") else ""
        print(f"    {name}: F1={f1_c:.4f}{flag}")

    print("\n  Full Classification Report:")
    print(classification_report(
        y_test, test_preds,
        target_names=CLASS_NAMES,
        zero_division=0, digits=4
    ))

    # ── 6. SHAP pilot ──
    print("\n[6] SHAP GradientExplainer pilot (5 real samples from inter_test)...")

    # Use real samples, not random noise — more representative
    rng = np.random.default_rng(42)
    pilot_idx = rng.choice(len(X_test), 5, replace=False)
    test_x    = torch.from_numpy(X_test[pilot_idx])
    bg_idx    = rng.choice(len(X_test), 50, replace=False)
    bg        = torch.from_numpy(X_test[bg_idx])

    class ModelWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x)

    wrapper = ModelWrapper(model).eval()
    try:
        explainer = shap.GradientExplainer(wrapper, bg)
        shap_vals = explainer.shap_values(test_x)

        print(f"  ✓ SHAP GradientExplainer SUCCESS")
        print(f"    Output type:  {type(shap_vals)}")
        print(f"    Output shape: {np.array(shap_vals).shape}")
        print(f"    (expected: [5 classes, 5 samples, 187 timesteps, 1 channel])")

        # Sanity: SHAP values should be non-trivial
        abs_max = np.abs(np.array(shap_vals)).max()
        if abs_max < 1e-8:
            print(f"  WARNING: SHAP values near zero (max={abs_max:.2e}) — potential gradient masking")
        else:
            print(f"    Max |SHAP|: {abs_max:.4f} — non-trivial ✓")

    except Exception as e:
        print(f"  ✗ SHAP FAILED: {e}")
        print("    → Must switch to alternative explainer before running T2/T3")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Checkpoint:    {ckpt_path}")
    print(f"  Hash:          {ckpt_hash}")
    print(f"  Best epoch:    {saved_epoch}")
    print(f"  Val F1 (ckpt): {saved_val_f1:.4f}")
    print(f"  Val F1 (now):  {val_f1:.4f}  ✓" if delta_f1 <= 0.02 else f"  Val F1 mismatch! {val_f1:.4f} vs {saved_val_f1:.4f}")
    print(f"  Test Macro F1: {test_f1_macro:.4f}")
    print(f"  SHAP pilot:    see above")
    print("\n  ✓ Ready to run T1b / T2 / T3 / T-NEW" if delta_f1 <= 0.02 else
          "\n  ✗ Resolve checkpoint mismatch before proceeding")


if __name__ == "__main__":
    main()
