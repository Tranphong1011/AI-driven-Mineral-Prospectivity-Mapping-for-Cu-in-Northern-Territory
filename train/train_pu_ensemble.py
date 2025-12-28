# train_pu_ensemble.py
# PU/background sampling for copper prospectivity
# Settings:
#  - pos1_bg1: 1:1 (random BG) -> 10 repeats
#  - pos1_bg2: 1:2 (random BG) -> 10 repeats
#  - all_bg:   use ALL background pool (no random BG) -> 1 run
#
# Metrics: PR-AUC + Recall@1% + Recall@5%
# Outputs: metrics CSV, per-repeat predictions, ensemble mean/std predictions

from __future__ import annotations

from dotenv import load_dotenv
import joblib
import shutil
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
import os
load_dotenv()

DIR = Path(os.getenv("BASE_DIR"))
BASE_DIR = Path(DIR / "Train_Final2")

DATA_TABLE = BASE_DIR / "grid_500m_ml_table.parquet"
FEATURE_COLS_TXT = BASE_DIR / "feature_cols.txt"

OUT_DIR = BASE_DIR / "ml_runs_pu_ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 3 settings
SETTINGS = {
    "pos1_bg1": {"bg_mode": "mult", "k_bg": 1, "repeats": 10},   # 1:1 random BG
    "pos1_bg2": {"bg_mode": "mult", "k_bg": 2, "repeats": 10},   # 1:2 random BG
    "all_bg":   {"bg_mode": "all",  "k_bg": None, "repeats": 1}, # ALL BG, no random
}

# Validation strategy:
VAL_FRAC = 0.2
BLOCK_SIZE_M = 20000  # 20 km blocks (tune if you want)
RANDOM_SEED_SPLIT = 1337

# Prediction chunk size to avoid RAM blow-ups
PRED_CHUNK = 200_000

# --- Save artifacts for XAI / reproducibility ---
SAVE_MODELS = True
SAVE_TRAIN_IDXS = True


BEST_METRIC = "pr_auc"   
BEST_SETTING_FOR_XAI = "pos1_bg2" 
XAI_BG_SAMPLE = 20000    
XAI_TOP_TARGETS = 500    



def load_feature_cols(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def recall_at_k_percent(y_true: np.ndarray, y_score: np.ndarray, k_percent: float) -> float:
    """Recall@K% over the provided validation set."""
    pos = y_true.sum()
    if pos == 0:
        return np.nan
    n = len(y_true)
    k = max(1, int(np.ceil(n * (k_percent / 100.0))))
    idx = np.argpartition(y_score, -k)[-k:]
    return float(y_true[idx].sum() / pos)


def make_block_split(df: pd.DataFrame, val_frac: float, block_size_m: int):
    """Spatial split using block ids from (cell_x, cell_y)."""
    bx = (df["cell_x"].values // block_size_m).astype(np.int64)
    by = (df["cell_y"].values // block_size_m).astype(np.int64)
    h = (bx * 73856093) ^ (by * 19349663)
    h = np.abs(h) % 100
    threshold = int(round(val_frac * 100))
    return h < threshold


def make_random_split(n: int, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    return rng.random(n) < val_frac


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["label", "has_any_ozmin_5km"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if "y" not in df.columns:
        df["y"] = (df["label"] > 0).astype(int)

    if "sample_weight" not in df.columns:
        df["sample_weight"] = 1.0
        pos_mask = df["y"] == 1
        df.loc[pos_mask, "sample_weight"] = df.loc[pos_mask, "label"].astype(float)

    return df



def main():
    feature_cols = load_feature_cols(FEATURE_COLS_TXT)

    print("Loading data table (big)...")
    df = pd.read_parquet(DATA_TABLE)
    df = ensure_columns(df)

    # Feature columns present & numeric
    feature_cols_present = [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not feature_cols_present:
        raise ValueError("No numeric feature columns found. Check feature_cols.txt.")

    print(f"Rows: {len(df):,}")
    print(f"Features: {len(feature_cols_present)}")

    # Masks
    y_all = df["y"].values.astype(int)
    pos_mask_all = (y_all == 1)
    bg_pool_mask_all = (y_all == 0) & (df["has_any_ozmin_5km"].values == 1)

    print(f"Positives (y=1): {pos_mask_all.sum():,}")
    print(f"BG pool (y=0 & has_any_ozmin_5km=1): {bg_pool_mask_all.sum():,}")

    # Split train/val
    use_spatial = ("cell_x" in df.columns) and ("cell_y" in df.columns) and df["cell_x"].notna().any() and df["cell_y"].notna().any()

    if use_spatial:
        print("Using BLOCK spatial split.")
        is_val = make_block_split(df, VAL_FRAC, BLOCK_SIZE_M)
    else:
        print("cell_x/cell_y missing -> using RANDOM split.")
        is_val = make_random_split(len(df), VAL_FRAC, RANDOM_SEED_SPLIT)

    is_train = ~is_val

    # Fixed validation set (positives + sampled background from bg pool in val)
    pos_val_idx = np.where(is_val & pos_mask_all)[0]
    bg_val_pool_idx = np.where(is_val & bg_pool_mask_all)[0]

    rng_val = np.random.default_rng(2025)
    n_pos_val = len(pos_val_idx)
    n_bg_val = min(len(bg_val_pool_idx), max(10 * n_pos_val, 50_000))
    if n_bg_val > 0:
        bg_val_idx = rng_val.choice(bg_val_pool_idx, size=n_bg_val, replace=False)
        val_idx = np.concatenate([pos_val_idx, bg_val_idx])
    else:
        val_idx = pos_val_idx

    y_val = y_all[val_idx].astype(int)

    # Save val info
    val_pack = {
        "val_idx_len": int(len(val_idx)),
        "val_pos": int(y_val.sum()),
        "val_bg": int((y_val == 0).sum()),
        "use_spatial": bool(use_spatial),
        "VAL_FRAC": VAL_FRAC,
        "BLOCK_SIZE_M": BLOCK_SIZE_M,
    }
    (OUT_DIR / "val_info.json").write_text(json.dumps(val_pack, indent=2), encoding="utf-8")
    np.save(OUT_DIR / "val_idx.npy", val_idx)
    np.save(OUT_DIR / "y_val.npy", y_val)
    (OUT_DIR / "run_config.json").write_text(
        json.dumps(
            {
                "settings": SETTINGS,
                "features": feature_cols_present,
                "model": "HistGradientBoostingClassifier",
                "VAL_FRAC": VAL_FRAC,
                "BLOCK_SIZE_M": BLOCK_SIZE_M,
                "PRED_CHUNK": PRED_CHUNK,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Validation set:", val_pack)

    # Pre-extract X_val
    X_val = df.loc[val_idx, feature_cols_present].to_numpy(dtype=np.float32, copy=False)

    N = len(df)

    # Train pools restricted to TRAIN partition
    pos_train_idx_all = np.where(is_train & pos_mask_all)[0]
    bg_train_pool_idx_all = np.where(is_train & bg_pool_mask_all)[0]
    n_pos_train = len(pos_train_idx_all)
    n_bg_train_pool = len(bg_train_pool_idx_all)

    print(f"Train positives: {n_pos_train:,}")
    print(f"Train BG pool: {n_bg_train_pool:,}")

    all_metrics_rows = []

    for setting_name, cfg in SETTINGS.items():
        setting_dir = OUT_DIR / setting_name
        setting_dir.mkdir(parents=True, exist_ok=True)

        val_probs_dir = setting_dir / "val_probs"
        pred_dir = setting_dir / "pred_full"
        val_probs_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

        repeats = int(cfg["repeats"])
        bg_mode = cfg["bg_mode"]

        print(f"\n=== Setting: {setting_name} | mode={bg_mode} | repeats={repeats} ===")
        best_score = -np.inf
        best_repeat = None
        best_model_path = None
        for r in range(repeats):
            # For all_bg, we keep deterministic and run once
            seed = 1000 + r
            rng = np.random.default_rng(seed)

            # Choose BG indices
            if bg_mode == "mult":
                k_bg = int(cfg["k_bg"])
                n_bg_needed = k_bg * n_pos_train
                if n_bg_needed > n_bg_train_pool:
                    print(f"WARNING: Not enough BG for {setting_name}. Using full BG pool.")
                    chosen_bg = bg_train_pool_idx_all
                else:
                    chosen_bg = rng.choice(bg_train_pool_idx_all, size=n_bg_needed, replace=False)

            elif bg_mode == "all":
                chosen_bg = bg_train_pool_idx_all  # ALL BG, no random
                n_bg_needed = len(chosen_bg)
            else:
                raise ValueError("Unknown bg_mode")

            train_idx = np.concatenate([pos_train_idx_all, chosen_bg])

            # Build train arrays
            X_train = df.loc[train_idx, feature_cols_present].to_numpy(dtype=np.float32, copy=False)
            y_train = y_all[train_idx].astype(int)
            w_train = df["sample_weight"].values[train_idx].astype(np.float32)

            model = HistGradientBoostingClassifier(
                max_depth=8,
                learning_rate=0.07,
                max_iter=300,
                min_samples_leaf=50,
                l2_regularization=0.0,
                random_state=seed,
            )

            print(f"Training {setting_name} repeat {r+1}/{repeats}... train_n={len(train_idx):,}")
            model.fit(X_train, y_train, sample_weight=w_train)


            # --- Save model + indices for reproducibility ---
            if SAVE_MODELS:
                models_dir = setting_dir / "models"
                models_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, models_dir / f"model_repeat_{r:02d}.joblib")

            if SAVE_TRAIN_IDXS:
                idx_dir = setting_dir / "train_indices"
                idx_dir.mkdir(parents=True, exist_ok=True)
                np.save(idx_dir / f"pos_train_idx.npy", pos_train_idx_all)  # fixed across repeats
                np.save(idx_dir / f"bg_chosen_repeat_{r:02d}.npy", chosen_bg.astype(np.int64))
                np.save(idx_dir / f"train_idx_repeat_{r:02d}.npy", train_idx.astype(np.int64))


            # Validate
            val_prob = model.predict_proba(X_val)[:, 1].astype(np.float32)
            pr_auc = float(average_precision_score(y_val, val_prob))
            r1 = recall_at_k_percent(y_val, val_prob, 1.0)
            r5 = recall_at_k_percent(y_val, val_prob, 5.0)


            # --- choose best repeat for this setting ---
            metric_map = {"pr_auc": pr_auc, "recall_at_1pct": r1, "recall_at_5pct": r5}
            score_now = float(metric_map[BEST_METRIC])

            if score_now > best_score:
                best_score = score_now
                best_repeat = r
                if SAVE_MODELS:
                    best_model_path = (setting_dir / "models" / f"model_repeat_{r:02d}.joblib")


            np.save(val_probs_dir / f"val_prob_repeat_{r:02d}.npy", val_prob)

            # Predict full grid
            pred_path = pred_dir / f"pred_full_repeat_{r:02d}.npy"
            pred_full = np.memmap(pred_path, dtype="float32", mode="w+", shape=(N,))

            for start in range(0, N, PRED_CHUNK):
                end = min(N, start + PRED_CHUNK)
                X_chunk = df.iloc[start:end][feature_cols_present].to_numpy(dtype=np.float32, copy=False)
                pred_full[start:end] = model.predict_proba(X_chunk)[:, 1].astype(np.float32)

            pred_full.flush()
            del pred_full

            all_metrics_rows.append(
                {
                    "setting": setting_name,
                    "bg_mode": bg_mode,
                    "k_bg": cfg.get("k_bg", None),
                    "repeat": r,
                    "seed": seed,
                    "train_pos": int(n_pos_train),
                    "train_bg": int(n_bg_needed),
                    "val_n": int(len(val_idx)),
                    "val_pos": int(y_val.sum()),
                    "val_bg": int((y_val == 0).sum()),
                    "pr_auc": pr_auc,
                    "recall_at_1pct": r1,
                    "recall_at_5pct": r5,
                    "pred_file": str(pred_path),
                    "val_prob_file": str(val_probs_dir / f"val_prob_repeat_{r:02d}.npy"),
                }
            )

            del X_train, y_train, w_train

        # --- Save best model link/copy for this setting ---
        if SAVE_MODELS and best_model_path is not None:
            dst = setting_dir / "best_model.joblib"
            shutil.copy2(best_model_path, dst)
            meta = {"best_metric": BEST_METRIC, "best_score": best_score, "best_repeat": int(best_repeat)}
            (setting_dir / "best_model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Aggregate predictions mean/std for this setting
        print(f"Aggregating predictions for {setting_name}...")
        pred_files = sorted(pred_dir.glob("pred_full_repeat_*.npy"))
        if len(pred_files) != repeats:
            raise RuntimeError(f"Expected {repeats} pred files, found {len(pred_files)}")

        sum_pred = np.zeros(N, dtype=np.float64)
        sumsq_pred = np.zeros(N, dtype=np.float64)

        for fp in pred_files:
            arr = np.memmap(fp, dtype="float32", mode="r", shape=(N,))
            sum_pred += arr
            sumsq_pred += arr.astype(np.float64) ** 2

        mean_pred = (sum_pred / repeats).astype(np.float32)
        var_pred = (sumsq_pred / repeats) - (mean_pred.astype(np.float64) ** 2)
        var_pred = np.maximum(var_pred, 0.0)
        std_pred = np.sqrt(var_pred).astype(np.float32)

        np.save(setting_dir / "pred_mean.npy", mean_pred)
        np.save(setting_dir / "pred_std.npy", std_pred)

        pd.DataFrame({"pred_mean": mean_pred, "pred_std": std_pred}).to_parquet(
            setting_dir / "pred_mean_std.parquet", index=False
        )

        print(f"Saved: {setting_dir / 'pred_mean_std.parquet'}")

    # Save metrics
    metrics_df = pd.DataFrame(all_metrics_rows)
    metrics_df.to_csv(OUT_DIR / "metrics_all_repeats.csv", index=False)

    # Summary (mean/std) by setting
    summary = (
        metrics_df.groupby("setting")
        .agg(
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            r1_mean=("recall_at_1pct", "mean"),
            r1_std=("recall_at_1pct", "std"),
            r5_mean=("recall_at_5pct", "mean"),
            r5_std=("recall_at_5pct", "std"),
            repeats=("repeat", "count"),
            train_bg_mean=("train_bg", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "metrics_summary_by_setting.csv", index=False)

    print("\nDone. Summary:")
    print(summary)

    xai_dir = OUT_DIR / "xai_pack"
    xai_dir.mkdir(parents=True, exist_ok=True)

    # save feature cols for SHAP scripts
    (xai_dir / "feature_cols.json").write_text(json.dumps(feature_cols_present, indent=2), encoding="utf-8")

    # save val split artifacts (already saved, but copy here for convenience)
    shutil.copy2(OUT_DIR / "val_idx.npy", xai_dir / "val_idx.npy")
    shutil.copy2(OUT_DIR / "y_val.npy", xai_dir / "y_val.npy")
    shutil.copy2(OUT_DIR / "val_info.json", xai_dir / "val_info.json")

    # copy best model from best setting
    best_setting_dir = OUT_DIR / BEST_SETTING_FOR_XAI
    best_model_file = best_setting_dir / "best_model.joblib"
    if best_model_file.exists():
        shutil.copy2(best_model_file, xai_dir / "best_model.joblib")
        shutil.copy2(best_setting_dir / "best_model_meta.json", xai_dir / "best_model_meta.json")

    # make SHAP background indices (sample from TRAIN bg pool)
    rng = np.random.default_rng(999)
    bg_train_pool = bg_train_pool_idx_all
    n_bg = min(len(bg_train_pool), XAI_BG_SAMPLE)
    bg_shap_idx = rng.choice(bg_train_pool, size=n_bg, replace=False).astype(np.int64)
    np.save(xai_dir / "bg_shap_idx.npy", bg_shap_idx)

    # choose top targets indices from ensemble mean (pos1_bg2)
    mean_pred_path = best_setting_dir / "pred_mean.npy"
    if mean_pred_path.exists():
        mean_pred = np.load(mean_pred_path).astype(np.float32)

        # explain top targets only among unlabelled/background (y==0)
        unlab_idx = np.where(y_all == 0)[0]
        k = min(XAI_TOP_TARGETS, len(unlab_idx))
        # fast top-k
        topk_part = np.argpartition(mean_pred[unlab_idx], -k)[-k:]
        topk_idx = unlab_idx[topk_part]
        # sort descending for nicer reports
        topk_idx = topk_idx[np.argsort(mean_pred[topk_idx])[::-1]].astype(np.int64)

        np.save(xai_dir / "top_targets_idx.npy", topk_idx)

    # write meta
    meta = {
        "best_setting": BEST_SETTING_FOR_XAI,
        "best_metric": BEST_METRIC,
        "xai_bg_sample": int(XAI_BG_SAMPLE),
        "xai_top_targets": int(XAI_TOP_TARGETS),
    }
    (xai_dir / "xai_pack_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved XAI pack to:", xai_dir)


if __name__ == "__main__":
    main()
