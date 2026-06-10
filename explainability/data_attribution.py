"""
T-NEW — Data Attribution: TracIn Influence Functions (Single Checkpoint Approximation).

Influence(z_train, z_test) ≈ cosine_sim(grad_loss(z_train), grad_loss(z_test))

Terminology (deliberately conservative):
  "harmful/confusable training samples" — any sample with high harmful influence.
  "label noise candidate" — subset that ALSO passes train-disagreement check:
      model self-predicts a DIFFERENT class from the annotated label with high
      confidence (self_pred != train_label AND self_conf > 0.9).
  Only these two-signal candidates are labeled "noise" in the paper.

Why NOT F1-flip as verification:
  5–20 candidates in 42K training samples → ΔF1 ≈ 0 regardless of ground truth.
  The test is statistically underpowered and produces a false sense of verification.
  Removed. Camera-ready option: scale to 100s via self-influence full-train sweep +
  fine-tune classifier head only (BN frozen) and measure minority F1 delta.

Why cosine similarity instead of raw dot product:
  Raw dot product is dominated by gradient norm differences across samples.
  Cosine normalizes magnitude, ranking purely by gradient direction alignment.
  We report cosine as primary and optionally log raw for supplementary.

Why gradient over classifier params only:
  classifier.0 (Linear 192→96) + classifier.3 (Linear 96→5) ≈ 19K params.
  Full network ≈ 505K params; 96% is BiLSTM feature encoder whose gradients
  encode representation changes, not decision boundary changes.
  Decision-boundary gradient is what TracIn requires.

Outputs:
  outputs/<run_id>/explainability/
    tracin_top_harmful.json         ← per-test top harmful samples + self-disagreement
    label_noise_candidates.csv      ← two-signal confirmed candidates only
    tracin_waveforms/               ← waveform pair PNG for top-30 confusable samples
"""

import json
import csv
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.paths import get_run_id, build_paths, RLSTM_CKPT, get_checkpoint_hash, INTER_TRAIN, INTER_TEST
from report_results import load_hmr_bilstm

CLASS_NAMES  = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}
AAMI_MAPPING = {
    0: "Normal (N, L, R, e, j)",
    1: "Supraventricular (A, a, J, S)",
    2: "Ventricular (V, E)",
    3: "Fusion (F)",
    4: "Unknown (/, f, Q)",
}


# ─── Gradient extraction ───────────────────────────────────────────────────────

def get_classifier_gradient(model, x, y, criterion, device):
    """
    Per-sample gradient w.r.t. classifier parameters (classifier.0 + classifier.3).

    model.eval():
      - BatchNorm uses frozen running stats → gradient is deterministic
        regardless of what other samples are in the batch (none, here)
      - Dropout is off → consistent gradient magnitudes across samples
    Both properties are essential for reliable influence ranking.
    """
    model.eval()
    model.zero_grad()

    x = x.to(device)
    y = y.to(device)

    with torch.enable_grad():
        logits = model(x)
        loss, _ = criterion(logits, y, r_fwd=None, r_bwd=None)
        loss.backward()

    grads = []
    for name, param in model.named_parameters():
        if "classifier" in name and param.grad is not None:
            grads.append(param.grad.detach().view(-1).cpu())

    # Safety fallback — should not trigger with HMR-BiLSTM
    if not grads:
        raise RuntimeError(
            "No 'classifier' parameters found with gradients. "
            "Check model parameter naming."
        )

    model.zero_grad()
    return torch.cat(grads)


# ─── Self-disagreement check ───────────────────────────────────────────────────

@torch.no_grad()
def check_train_disagreement(model, x_train_sample, train_label_int, device,
                              confidence_threshold=0.9):
    """
    Model self-disagreement: does the model predict a DIFFERENT class
    for this training sample from its annotated label, with high confidence?

    Two independent signals confirming noise:
      1. TracIn: sample is harmful (pushes model toward wrong prediction)
      2. Self-disagreement: model itself rejects the sample's label

    A candidate with both signals is a much stronger noise claim than either alone.
    """
    x = torch.from_numpy(x_train_sample).unsqueeze(0).to(device)
    probs = model(x).softmax(-1)[0]
    self_pred = int(probs.argmax())
    self_conf = float(probs.max())
    agrees_noise = (self_pred != train_label_int) and (self_conf > confidence_threshold)
    return {
        "model_self_pred":  CLASS_NAMES.get(self_pred, str(self_pred)),
        "model_self_conf":  round(self_conf, 4),
        "agrees_noise":     agrees_noise,
        "disagreement_str": (
            f"model predicts {CLASS_NAMES.get(self_pred)} (conf={self_conf:.2f}) "
            f"≠ annotated {CLASS_NAMES.get(train_label_int)}"
            if agrees_noise else "no disagreement"
        ),
    }


# ─── Sample selection ──────────────────────────────────────────────────────────

def select_misclassified_stratified(y_test, preds, rng, num_samples=20):
    """
    Stratified selection prioritising minority AAMI classes (S=1, V=2, F=3).
    These are clinically critical and more likely to expose real noise.
    """
    misclassified_idx = np.where(y_test != preds)[0]
    if len(misclassified_idx) <= num_samples:
        return misclassified_idx

    priority_order = [1, 2, 3, 0, 4]  # S, V, F first
    selected = []
    budget   = num_samples

    present_classes = [c for c in priority_order
                       if np.any((y_test == c) & (preds != c))]
    quota = max(2, budget // max(1, len(present_classes)))

    for c in priority_order:
        if budget <= 0:
            break
        c_idx = np.where((y_test == c) & (preds != c))[0]
        if len(c_idx) == 0:
            continue
        n_sel = min(quota, len(c_idx), budget)
        selected.extend(rng.choice(c_idx, n_sel, replace=False))
        budget -= n_sel

    return np.array(list(dict.fromkeys(selected)))  # dedup, preserve order


# ─── Waveform visualisation ────────────────────────────────────────────────────

def plot_waveform_pair(train_wave, train_label_int, train_idx,
                       test_wave, test_true_int, test_pred_int, test_idx,
                       influence_val, self_disagree_info, out_path):
    """
    Side-by-side waveform for visual inspection of whether the training sample
    morphology matches its annotated AAMI class.
    Title encodes all signals needed for reviewer: label, self-pred, influence.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor("#0f1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1d2e")
        ax.tick_params(colors="#aaa")
        for sp in ax.spines.values():
            sp.set_color("#555")

    t = np.arange(len(train_wave.squeeze()))

    # Signal: red border if self-disagreement confirms noise
    edge_color = "#ef5350" if self_disagree_info["agrees_noise"] else "#78909c"
    ax1.plot(t, train_wave.squeeze(), color=edge_color, lw=1.5)
    noise_flag = "⚠ SELF-DISAGREE" if self_disagree_info["agrees_noise"] else ""
    ax1.set_title(
        f"TRAIN #{train_idx}  Label: {CLASS_NAMES[train_label_int]}\n"
        f"({AAMI_MAPPING[train_label_int]})\n"
        f"Model sees: {self_disagree_info['model_self_pred']} "
        f"(conf={self_disagree_info['model_self_conf']:.2f})  {noise_flag}",
        color="white", fontsize=9
    )
    ax1.set_xlabel("Timestep (125 Hz)", color="#aaa")
    ax1.set_ylabel("Amplitude (Z-score)", color="#aaa")
    ax1.grid(True, linestyle="--", alpha=0.3, color="#555")

    ax2.plot(t, test_wave.squeeze(), color="#42a5f5", lw=1.5)
    ax2.set_title(
        f"TEST #{test_idx}\n"
        f"True: {CLASS_NAMES[test_true_int]}  Pred: {CLASS_NAMES[test_pred_int]}\n"
        f"Cosine influence: {influence_val:.4f}",
        color="white", fontsize=9
    )
    ax2.set_xlabel("Timestep (125 Hz)", color="#aaa")
    ax2.grid(True, linestyle="--", alpha=0.3, color="#555")

    plt.suptitle(
        "TracIn Confusable Sample — visual noise inspection",
        color="white", fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    config_path = Path("configs/experiment_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_id  = get_run_id(cfg)
    paths   = build_paths(run_id)
    out_dir  = paths["out_explain"]
    wave_dir = out_dir / "tracin_waveforms"
    out_dir.mkdir(parents=True, exist_ok=True)
    wave_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.get("seed", 42)
    rng  = np.random.default_rng(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TracIn] Device: {device} | Run ID: {run_id}")

    # ── Load model ──
    print("Loading model...")
    model, _ = load_hmr_bilstm(RLSTM_CKPT, device)
    model.eval()

    from hmr_bilstm import RLSTMLoss
    # lambda_smooth=0.0: attribute only to classification loss
    criterion = RLSTMLoss(lambda_smooth=0.0).to(device)

    # ── Load data ──
    print("Loading data...")
    test_file  = INTER_TEST
    train_file = INTER_TRAIN

    test    = np.load(test_file)
    X_test  = test["X"].astype(np.float32)
    y_test  = test["y"].astype(np.int64)

    train   = np.load(train_file)
    X_train = train["X"].astype(np.float32)
    y_train = train["y"].astype(np.int64)
    print(f"  Test: {len(X_test)} | Train: {len(X_train)}")

    # ── Stratified train subset ──
    n_train_subset = 5000
    if len(X_train) > n_train_subset:
        sub_list = []
        for c in np.unique(y_train):
            c_idx = np.where(y_train == c)[0]
            n_c   = min(n_train_subset // len(np.unique(y_train)), len(c_idx))
            sub_list.extend(rng.choice(c_idx, n_c, replace=False))
        rem = n_train_subset - len(sub_list)
        if rem > 0:
            pool = list(set(range(len(X_train))) - set(sub_list))
            sub_list.extend(rng.choice(pool, min(rem, len(pool)), replace=False))
        train_idx = np.array(sub_list)
    else:
        train_idx = np.arange(len(X_train))

    X_train_sub = X_train[train_idx]
    y_train_sub = y_train[train_idx]
    print(f"  Training subset: {len(X_train_sub)} (stratified)")

    # ── Test predictions ──
    print("Getting test predictions...")
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 256):
            b = torch.from_numpy(X_test[i:i+256]).to(device)
            all_preds.append(model(b).argmax(dim=-1).cpu().numpy())
    preds_all = np.concatenate(all_preds)
    acc = (preds_all == y_test).mean()
    print(f"  Test accuracy: {acc:.4f}")

    # ── Select misclassified ──
    misclassified = select_misclassified_stratified(y_test, preds_all, rng, num_samples=20)
    print(f"  Selected {len(misclassified)} misclassified samples for attribution")

    # ── Pre-compute TRAIN gradients (cosine-normalised) ──
    print("Pre-computing training gradients (classifier params, cosine-normalised)...")
    train_grads = []
    for i in range(len(X_train_sub)):
        if i % 500 == 0:
            print(f"  [{i}/{len(X_train_sub)}]")
        x_tr = torch.from_numpy(X_train_sub[i:i+1])
        y_tr = torch.tensor([y_train_sub[i]])
        g = get_classifier_gradient(model, x_tr, y_tr, criterion, device)
        train_grads.append(g)

    train_grads_raw = torch.stack(train_grads)               # (N, D)
    train_grads_cos = F.normalize(train_grads_raw, dim=1)    # unit-norm rows
    print(f"  Gradient matrix: {train_grads_cos.shape} (cosine-normalised)")

    # ── Compute influence ──
    print("Computing TracIn cosine influence scores...")
    results          = []
    confusable_list  = []   # all "harmful/confusable" samples (criterion 1+2)
    noise_confirmed  = []   # subset also passing self-disagreement (criterion 3)

    for idx in misclassified:
        x_te    = torch.from_numpy(X_test[idx:idx+1])
        y_te    = torch.tensor([y_test[idx]])
        pred_te = int(preds_all[idx])
        true_te = int(y_test[idx])

        test_grad_raw = get_classifier_gradient(model, x_te, y_te, criterion, device)
        test_grad_cos = F.normalize(test_grad_raw, dim=0)   # unit norm

        # Cosine similarity: (N,) values in [-1, 1]
        cosine_inf = (train_grads_cos @ test_grad_cos).numpy()

        # Top-5 most harmful (most negative cosine)
        top5_rel = np.argsort(cosine_inf)[:5]
        top5_abs = train_idx[top5_rel]

        harmful_samples = []
        for rel_i, abs_i in zip(top5_rel, top5_abs):
            tr_label = int(y_train_sub[rel_i])
            inf_val  = float(cosine_inf[rel_i])

            # Self-disagreement check for this training sample
            self_info = check_train_disagreement(
                model, X_train[abs_i], tr_label, device
            )

            harmful_samples.append({
                "train_index":       int(abs_i),
                "train_label":       CLASS_NAMES[tr_label],
                "cosine_influence":  round(inf_val, 6),
                **self_info,
            })

            # Criterion for confusable: train_label == wrong_pred
            if tr_label == pred_te and pred_te != true_te:
                entry = {
                    "test_idx":         int(idx),
                    "test_true":        CLASS_NAMES[true_te],
                    "test_true_int":    true_te,
                    "test_pred":        CLASS_NAMES[pred_te],
                    "test_pred_int":    pred_te,
                    "train_idx":        int(abs_i),
                    "train_label":      CLASS_NAMES[tr_label],
                    "train_label_int":  tr_label,
                    "cosine_influence": round(inf_val, 6),
                    **self_info,
                }
                confusable_list.append(entry)
                if self_info["agrees_noise"]:
                    noise_confirmed.append(entry)

        results.append({
            "test_index":               int(idx),
            "true_label":               CLASS_NAMES[true_te],
            "predicted_label":          CLASS_NAMES[pred_te],
            "top_harmful_train_samples": harmful_samples,
        })

    confusable_list.sort(key=lambda e: e["cosine_influence"])
    noise_confirmed.sort(key=lambda e: e["cosine_influence"])

    print(f"  Confusable samples (train_label==wrong_pred): {len(confusable_list)}")
    print(f"  Confirmed noise (+ self-disagreement >0.9):   {len(noise_confirmed)}")

    # ── Waveform visualisation (top-30 confusable) ──
    print(f"Generating waveform plots for top-{min(30, len(confusable_list))} confusable samples...")
    for i, cand in enumerate(confusable_list[:30]):
        t_idx  = cand["train_idx"]
        te_idx = cand["test_idx"]
        flag   = "_NOISE" if cand["agrees_noise"] else ""
        fname  = (
            f"cand_{i:02d}{flag}_"
            f"tr{t_idx}_{cand['train_label']}_"
            f"te{te_idx}_{cand['test_true']}pred{cand['test_pred']}.png"
        )
        plot_waveform_pair(
            train_wave        = X_train[t_idx],
            train_label_int   = cand["train_label_int"],
            train_idx         = t_idx,
            test_wave         = X_test[te_idx],
            test_true_int     = cand["test_true_int"],
            test_pred_int     = cand["test_pred_int"],
            test_idx          = te_idx,
            influence_val     = cand["cosine_influence"],
            self_disagree_info= {k: cand[k] for k in
                                  ("model_self_pred","model_self_conf",
                                   "agrees_noise","disagreement_str")},
            out_path          = wave_dir / fname,
        )
    print(f"  Saved to {wave_dir}/")

    # ── Save main results JSON ──
    out_json = {
        "experiment_version":   cfg["experiment"]["version"],
        "run_id":               run_id,
        "checkpoint_hash":      get_checkpoint_hash(RLSTM_CKPT),
        "module":               "data_attribution",
        "timestamp":            datetime.now().isoformat(),
        "methodology": {
            "influence_metric":       "cosine similarity (F.normalize grad vectors)",
            "gradient_scope":         "classifier parameters only (~19K)",
            "noise_candidate_criteria": [
                "criterion_1: top-5 harmful cosine influence for a misclassified test sample",
                "criterion_2: train_label == model's wrong prediction (confusable)",
                "criterion_3 (confirmed noise): model self-pred != train_label with conf > 0.9"
            ],
            "verification":           "train-disagreement (two independent signals)",
            "f1_flip_note":           (
                "F1-flip NOT performed: 5-20 candidates in 42K training samples "
                "is statistically underpowered (expected ΔF1 ≈ 0). "
                "Camera-ready plan: self-influence full-train sweep → scale to 100s → "
                "fine-tune classifier head only (BN frozen) → measure minority F1 delta."
            )
        },
        "summary": {
            "n_misclassified_eval":  len(misclassified),
            "n_train_subset":        len(X_train_sub),
            "n_confusable":          len(confusable_list),
            "n_confirmed_noise":     len(noise_confirmed),
        },
        "tracin_results": results,
    }
    out_json_path = out_dir / "tracin_top_harmful.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"  [OK] {out_json_path}")

    # ── CSV: confirmed noise candidates only ──
    out_csv = out_dir / "label_noise_candidates.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "test_idx", "test_true", "test_pred",
            "train_idx", "train_label",
            "cosine_influence",
            "model_self_pred", "model_self_conf",
            "agrees_noise", "disagreement_str"
        ])
        for rank, c in enumerate(noise_confirmed, 1):
            writer.writerow([
                rank, c["test_idx"], c["test_true"], c["test_pred"],
                c["train_idx"], c["train_label"],
                f"{c['cosine_influence']:.6f}",
                c["model_self_pred"], f"{c['model_self_conf']:.4f}",
                c["agrees_noise"], c["disagreement_str"]
            ])
    print(f"  [OK] {out_csv}  ({len(noise_confirmed)} two-signal confirmed candidates)")

    # ── Also save full confusable list for reference ──
    confusable_csv = out_dir / "confusable_samples_all.csv"
    with open(confusable_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "test_idx", "test_true", "test_pred",
            "train_idx", "train_label", "cosine_influence",
            "model_self_pred", "model_self_conf", "agrees_noise"
        ])
        for rank, c in enumerate(confusable_list, 1):
            writer.writerow([
                rank, c["test_idx"], c["test_true"], c["test_pred"],
                c["train_idx"], c["train_label"],
                f"{c['cosine_influence']:.6f}",
                c["model_self_pred"], f"{c['model_self_conf']:.4f}",
                c["agrees_noise"]
            ])
    print(f"  [OK] {confusable_csv}  ({len(confusable_list)} confusable samples total)")

    print(f"\n[TracIn] Complete.")
    print(f"  → Visual: inspect {wave_dir}/ (files with _NOISE flag = two-signal confirmed)")
    print(f"  → Paper claim: {len(noise_confirmed)} training samples confirmed as "
          f"likely mislabeled by TracIn + model self-disagreement")


if __name__ == "__main__":
    main()
