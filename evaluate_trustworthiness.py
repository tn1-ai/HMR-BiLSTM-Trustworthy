import os
import json
import csv
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from hmr_bilstm import RLSTMClassifier
from configs.paths import get_run_id, build_paths, RLSTM_CKPT
from evaluate_fgsm import build_test_loader
from report_results import load_hmr_bilstm

def evaluate_classification(model, loader, device):
    """Evaluates classification accuracy, precision, recall, and F1 on the clean test loader."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision_macro": float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    }
    return metrics

def load_json_results(path: Path):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
            return None
    return None

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
    
    print(f"================================================================================")
    print(f" TRUSTWORTHY ECG EVALUATION DASHBOARD")
    print(f" Run ID: {run_id}")
    print(f"================================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Classification evaluation
    print("[1/5] Evaluating classification performance...")
    test_loader = build_test_loader(batch_size=128)
    
    if Path(RLSTM_CKPT).exists():
        try:
            model, _ = load_hmr_bilstm(RLSTM_CKPT, device)
            class_metrics = evaluate_classification(model, test_loader, device)
        except Exception as e:
            print(f"Error evaluating classification: {e}")
            class_metrics = {"accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}
    else:
        print(f"Warning: Checkpoint {RLSTM_CKPT} not found. Using default/zero values.")
        class_metrics = {"accuracy": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}
        
    print(f"  Classification metrics: F1-macro = {class_metrics['f1_macro']:.4f}, Accuracy = {class_metrics['accuracy']:.4f}")
    
    # Load module results
    calib_res = load_json_results(paths["out_calib"] / "results.json")
    explain_res = load_json_results(paths["out_explain"] / "results.json")
    
    # ── Load Uncertainty Results ──
    mc_res = load_json_results(paths["out_uncert"] / "mc_results.json")
    ens_res = load_json_results(paths["out_uncert"] / "ensemble_results.json")
    uncert_res = None
    if mc_res or ens_res:
        uncert_res = {"metrics": {}}
        if mc_res and "metrics" in mc_res:
            m_mc = mc_res["metrics"]
            uncert_res["metrics"]["mc_dropout"] = {
                "mean_entropy": m_mc.get("id_mean_entropy", 0.0),
                "mean_variance": m_mc.get("id_mean_mi", 0.0),
                "mean_confidence": m_mc.get("id_mean_conf", 0.0),
                "auroc_ood_mean": m_mc.get("ood_detection_auroc", 0.0)
            }
        if ens_res and "metrics" in ens_res:
            m_ens = ens_res["metrics"]
            uncert_res["metrics"]["deep_ensemble"] = {
                "mean_entropy": m_ens.get("id_mean_entropy", 0.0),
                "mean_variance": m_ens.get("id_mean_mi", 0.0),
                "mean_confidence": m_ens.get("id_mean_conf", 0.0),
                "auroc_ood_mean": m_ens.get("ood_detection_auroc", 0.0)
            }

    # ── Load Robustness Results ──
    fgsm_res_list = load_json_results(Path("results/logs/fgsm_baseline_comparison.json"))
    pgd_res_dict = load_json_results(Path("results/logs/pgd_baseline_comparison.json"))
    cw_res = load_json_results(paths["out_robust"] / "cw_attack_results.json")
    aa_res = load_json_results(paths["out_robust"] / "autoattack_results.json")
    
    robust_res = None
    if fgsm_res_list or pgd_res_dict or cw_res or aa_res:
        robust_res = {"metrics": {}}
        m = robust_res["metrics"]
        
        # Load Clean values from AA if possible
        if aa_res and "metrics" in aa_res:
            m["clean_acc"] = aa_res["metrics"].get("clean_accuracy", 0.0)
            m["clean_f1"] = aa_res["metrics"].get("clean_f1_macro", 0.0)
        elif cw_res and "metrics" in cw_res:
            m["clean_acc"] = class_metrics["accuracy"]
            m["clean_f1"] = cw_res["metrics"].get("clean_f1_macro", 0.0)
        else:
            m["clean_acc"] = class_metrics["accuracy"]
            m["clean_f1"] = class_metrics["f1_macro"]

        # Parse FGSM
        if fgsm_res_list:
            for item in fgsm_res_list:
                if item.get("model") == "HMR-BiLSTM" and item.get("epsilon") == 0.02:
                    m["fgsm_acc"] = item.get("accuracy", 0.0)
                    m["fgsm_f1"] = item.get("macro_f1", 0.0)
                    break
        
        # Parse PGD
        if pgd_res_dict and "HMR-BiLSTM" in pgd_res_dict:
            for item in pgd_res_dict["HMR-BiLSTM"]:
                if item.get("epsilon") == 0.02:
                    asr = item.get("attack_success_rate", 0.0)
                    m["pgd_acc"] = m["clean_acc"] * (1 - asr)
                    m["pgd_f1"] = item.get("macro_f1", 0.0)
                    break

        # Parse CW
        if cw_res and "metrics" in cw_res:
            m_cw = cw_res["metrics"]
            m["cw_acc"] = m["clean_acc"] * (1 - m_cw.get("asr_total", 0.0))
            m["cw_f1"] = m_cw.get("adv_f1_macro", 0.0)

        # Parse AutoAttack
        if aa_res and "metrics" in aa_res:
            m_aa = aa_res["metrics"]
            m["autoattack_acc"] = m["clean_acc"] * (1 - m_aa.get("autoattack_asr", 0.0))
            m["autoattack_f1"] = m_aa.get("autoattack_f1_macro", 0.0)
    
    # Create the output directory
    paths["out_root"].mkdir(parents=True, exist_ok=True)
    
    # --- TABLE 1: Classification Performance ---
    print("\n=== Table 1: Classification Performance ===")
    print(f"| Metric | Value |")
    print(f"|---|---|")
    print(f"| Accuracy | {class_metrics['accuracy']:.4f} |")
    print(f"| Precision (Macro) | {class_metrics['precision_macro']:.4f} |")
    print(f"| Recall (Macro) | {class_metrics['recall_macro']:.4f} |")
    print(f"| Macro F1 | {class_metrics['f1_macro']:.4f} |")
    
    # --- TABLE 2: Calibration ---
    print("\n=== Table 2: Calibration ===")
    print(f"| Metric | Before Temp Scaling | After Temp Scaling |")
    print(f"|---|---|---|")
    if calib_res and "metrics" in calib_res:
        m = calib_res["metrics"]
        print(f"| ECE | {m.get('ece_before', 0.0):.4f} | {m.get('ece_after', 0.0):.4f} |")
        print(f"| MCE | {m.get('mce_before', 0.0):.4f} | {m.get('mce_after', 0.0):.4f} |")
        print(f"| NLL | {m.get('nll_before', 0.0):.4f} | {m.get('nll_after', 0.0):.4f} |")
        print(f"| Brier Score | {m.get('brier_before', 0.0):.4f} | {m.get('brier_after', 0.0):.4f} |")
    else:
        print(f"| ECE | N/A | N/A |")
        print(f"| MCE | N/A | N/A |")
        print(f"| NLL | N/A | N/A |")
        print(f"| Brier Score | N/A | N/A |")
        
    # --- TABLE 3: Robustness ---
    print("\n=== Table 3: Robustness ===")
    print(f"| Attack | Accuracy | Macro F1 |")
    print(f"|---|---|---|")
    if robust_res and "metrics" in robust_res:
        m = robust_res["metrics"]
        print(f"| Clean | {m.get('clean_acc', 0.0):.4f} | {m.get('clean_f1', 0.0):.4f} |")
        print(f"| FGSM (e=0.02) | {m.get('fgsm_acc', 0.0):.4f} | {m.get('fgsm_f1', 0.0):.4f} |")
        print(f"| PGD (e=0.02) | {m.get('pgd_acc', 0.0):.4f} | {m.get('pgd_f1', 0.0):.4f} |")
        print(f"| CW (L2) | {m.get('cw_acc', 0.0):.4f} | {m.get('cw_f1', 0.0):.4f} |")
        print(f"| AutoAttack | {m.get('autoattack_acc', 0.0):.4f} | {m.get('autoattack_f1', 0.0):.4f} |")
    else:
        print(f"| Clean | {class_metrics['accuracy']:.4f} | {class_metrics['f1_macro']:.4f} |")
        print(f"| FGSM (e=0.02) | N/A | N/A |")
        print(f"| PGD (e=0.02) | N/A | N/A |")
        print(f"| CW (L2) | N/A | N/A |")
        print(f"| AutoAttack | N/A | N/A |")
        
    # --- TABLE 4: Uncertainty ---
    print("\n=== Table 4: Uncertainty ===")
    print(f"| Method | Mean Entropy | Mean Variance | Mean Confidence | AUROC-OOD (Mean) |")
    print(f"|---|---|---|---|---|")
    if uncert_res and "metrics" in uncert_res:
        m = uncert_res["metrics"]
        for method in ["mc_dropout", "deep_ensemble"]:
            met = m.get(method, {})
            print(f"| {method.replace('_', ' ').title()} | "
                  f"{met.get('mean_entropy', 0.0):.4f} | "
                  f"{met.get('mean_variance', 0.0):.4f} | "
                  f"{met.get('mean_confidence', 0.0):.4f} | "
                  f"{met.get('auroc_ood_mean', 0.0):.4f} |")
    else:
        print(f"| MC Dropout | N/A | N/A | N/A | N/A |")
        print(f"| Deep Ensemble | N/A | N/A | N/A | N/A |")
        
    # --- TABLE 5: Trustworthy Scorecard ---
    print("\n=== Table 5: Trustworthy Scorecard ===")
    print(f"| Dimension | Primary Metric | Score / Value |")
    print(f"|---|---|---|")
    print(f"| Accuracy | Macro F1 | {class_metrics['f1_macro']:.4f} |")
    
    ece_after_val = "N/A"
    if calib_res and "metrics" in calib_res:
        ece_after_val = f"{calib_res['metrics'].get('ece_after', 0.0):.4f}"
    print(f"| Calibration | ECE (after scaling) | {ece_after_val} |")
    
    shap_val = "N/A"
    if explain_res and "metrics" in explain_res:
        shap_val = explain_res["metrics"].get("shap_summary", "Present")
    print(f"| Explainability | SHAP top-3 features | {shap_val} |")
    
    aa_val = "N/A"
    if robust_res and "metrics" in robust_res:
        aa_val = f"{robust_res['metrics'].get('autoattack_f1', 0.0):.4f}"
    print(f"| Robustness | AutoAttack Macro F1 | {aa_val} |")
    
    ood_val = "N/A"
    if uncert_res and "metrics" in uncert_res:
        ood_val = f"{uncert_res['metrics'].get('mc_dropout', {}).get('auroc_ood_mean', 0.0):.4f}"
    print(f"| Uncertainty | MC Dropout AUROC-OOD | {ood_val} |")
    
    # Save files
    # 1. trustworthiness_summary.csv
    csv_path = paths["out_root"] / "trustworthiness_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Section", "Metric", "Value"])
        writer.writerow(["Classification", "Accuracy", f"{class_metrics['accuracy']:.4f}"])
        writer.writerow(["Classification", "Precision (Macro)", f"{class_metrics['precision_macro']:.4f}"])
        writer.writerow(["Classification", "Recall (Macro)", f"{class_metrics['recall_macro']:.4f}"])
        writer.writerow(["Classification", "Macro F1", f"{class_metrics['f1_macro']:.4f}"])
        if calib_res and "metrics" in calib_res:
            m = calib_res["metrics"]
            writer.writerow(["Calibration", "ECE Before", f"{m.get('ece_before', 0.0):.4f}"])
            writer.writerow(["Calibration", "ECE After", f"{m.get('ece_after', 0.0):.4f}"])
            writer.writerow(["Calibration", "MCE Before", f"{m.get('mce_before', 0.0):.4f}"])
            writer.writerow(["Calibration", "MCE After", f"{m.get('mce_after', 0.0):.4f}"])
            writer.writerow(["Calibration", "NLL Before", f"{m.get('nll_before', 0.0):.4f}"])
            writer.writerow(["Calibration", "NLL After", f"{m.get('nll_after', 0.0):.4f}"])
            writer.writerow(["Calibration", "Brier Before", f"{m.get('brier_before', 0.0):.4f}"])
            writer.writerow(["Calibration", "Brier After", f"{m.get('brier_after', 0.0):.4f}"])
        if robust_res and "metrics" in robust_res:
            m = robust_res["metrics"]
            writer.writerow(["Robustness", "Clean Acc", f"{m.get('clean_acc', 0.0):.4f}"])
            writer.writerow(["Robustness", "Clean F1", f"{m.get('clean_f1', 0.0):.4f}"])
            writer.writerow(["Robustness", "FGSM Acc", f"{m.get('fgsm_acc', 0.0):.4f}"])
            writer.writerow(["Robustness", "FGSM F1", f"{m.get('fgsm_f1', 0.0):.4f}"])
            writer.writerow(["Robustness", "PGD Acc", f"{m.get('pgd_acc', 0.0):.4f}"])
            writer.writerow(["Robustness", "PGD F1", f"{m.get('pgd_f1', 0.0):.4f}"])
            writer.writerow(["Robustness", "CW Acc", f"{m.get('cw_acc', 0.0):.4f}"])
            writer.writerow(["Robustness", "CW F1", f"{m.get('cw_f1', 0.0):.4f}"])
            writer.writerow(["Robustness", "AutoAttack Acc", f"{m.get('autoattack_acc', 0.0):.4f}"])
            writer.writerow(["Robustness", "AutoAttack F1", f"{m.get('autoattack_f1', 0.0):.4f}"])
        if uncert_res and "metrics" in uncert_res:
            m = uncert_res["metrics"]
            for method in ["mc_dropout", "deep_ensemble"]:
                met = m.get(method, {})
                writer.writerow(["Uncertainty", f"{method} Mean Entropy", f"{met.get('mean_entropy', 0.0):.4f}"])
                writer.writerow(["Uncertainty", f"{method} Mean Variance", f"{met.get('mean_variance', 0.0):.4f}"])
                writer.writerow(["Uncertainty", f"{method} Mean Confidence", f"{met.get('mean_confidence', 0.0):.4f}"])
                writer.writerow(["Uncertainty", f"{method} AUROC-OOD Mean", f"{met.get('auroc_ood_mean', 0.0):.4f}"])

    # 2. trustworthy_scorecard.csv
    scorecard_path = paths["out_root"] / "trustworthy_scorecard.csv"
    with open(scorecard_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Dimension", "Primary Metric", "Score / Value"])
        writer.writerow(["Accuracy", "Macro F1", f"{class_metrics['f1_macro']:.4f}"])
        writer.writerow(["Calibration", "ECE (after scaling)", ece_after_val])
        writer.writerow(["Explainability", "SHAP top-3 features", shap_val])
        writer.writerow(["Robustness", "AutoAttack Macro F1", aa_val])
        writer.writerow(["Uncertainty", "MC Dropout AUROC-OOD", ood_val])
        
    # 3. trustworthiness_summary.tex
    tex_path = paths["out_root"] / "trustworthiness_summary.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% LaTeX-ready tables for Trustworthy ECG\n")
        f.write("\\begin{table}[ht]\n\\centering\n\\caption{Trustworthy ECG Evaluation Scorecard}\n")
        f.write("\\begin{tabular}{llc}\n\\hline\n")
        f.write("Dimension & Primary Metric & Score/Value \\\\\n\\hline\n")
        f.write(f"Accuracy & Macro F1 & {class_metrics['f1_macro']:.4f} \\\\\n")
        f.write(f"Calibration & ECE (after scaling) & {ece_after_val} \\\\\n")
        f.write(f"Explainability & SHAP top-3 features & {shap_val} \\\\\n")
        f.write(f"Robustness & AutoAttack Macro F1 & {aa_val} \\\\\n")
        f.write(f"Uncertainty & MC Dropout AUROC-OOD & {ood_val} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
        
    print(f"\n[OK] Reports saved to:")
    print(f" - CSV summary: {csv_path}")
    print(f" - Scorecard: {scorecard_path}")
    print(f" - LaTeX table: {tex_path}")

if __name__ == "__main__":
    main()
