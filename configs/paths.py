import os
import yaml
from datetime import datetime
from pathlib import Path

def get_run_id(cfg: dict) -> str:
    """Sinh run_id từ version + timestamp nếu run_id == 'auto'."""
    # Hỗ trợ lấy run_id từ biến môi trường để chạy toàn bộ pipeline
    env_run_id = os.environ.get("TRUSTWORTHY_RUN_ID")
    if env_run_id:
        return env_run_id

    version = cfg["experiment"]["version"]          # "v1.0"
    run_id  = cfg["experiment"]["run_id"]            # "auto" hoặc tên cụ thể
    if run_id == "auto":
        if not hasattr(get_run_id, "_cached_id"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")  # "20260610_053012"
            get_run_id._cached_id = f"{version}_{ts}"
        return get_run_id._cached_id
    return run_id

def build_paths(run_id: str) -> dict:
    """Trả về dict đường dẫn cho một run cụ thể."""
    root = Path("outputs") / run_id
    return {
        "run_id":       run_id,
        "out_root":     root,
        "out_calib":    root / "calibration",
        "out_explain":  root / "explainability",
        "out_uncert":   root / "uncertainty",
        "out_robust":   root / "robustness",
        "out_figures":  root / "figures",
    }

def get_checkpoint_hash(ckpt_path: str) -> str:
    """Returns the last modified time and size of the checkpoint to uniquely identify it."""
    try:
        p = Path(ckpt_path)
        if p.exists():
            mtime = p.stat().st_mtime
            size = p.stat().st_size
            return f"mtime_{int(mtime)}_size_{size}"
        return "not_found"
    except Exception:
        return "error"

# Đường dẫn cố định (không phụ thuộc run)
RLSTM_CKPT     = "results/checkpoints/inter_best_rlstm.pt"
ENSEMBLE_DIR   = "results/checkpoints/ensemble"
DATA_PROCESSED = "data/processed"
INTER_TRAIN    = "data/processed/splits/inter_train.npz"
INTER_VAL      = "data/processed/splits/inter_val.npz"
INTER_TEST     = "data/processed/splits/inter_test.npz"

