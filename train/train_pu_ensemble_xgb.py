

from __future__ import annotations

import json
import shutil
from pathlib import Path
import os
from dotenv import load_dotenv
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
load_dotenv()

DIR = Path(os.getenv("BASE_DIR"))
BASE_DIR = Path(DIR / "Train_Final2")

DATA_TABLE = BASE_DIR / "grid_500m_ml_table.parquet"
FEATURE_COLS_TXT = BASE_DIR / "feature_cols.txt"

OUT_DIR = BASE_DIR / "ml_runs_pu_ensemble_xgb"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS = {
    "pos1_bg1": {"bg_mode": "mult", "k_bg": 1, "repeats": 10},
    "pos1_bg2": {"bg_mode": "mult", "k_bg": 2, "repeats": 10},
    "all_bg":   {"bg_mode": "all",  "k_bg": None, "repeats": 1},
}

VAL_FRAC = 0.2
BLOCK_SIZE_M = 20_000
RANDOM_SEED_SPLIT = 1337

PRED_CHUNK = 200_000

SAVE_MODELS = True
SAVE_TRAIN_IDXS = True

BEST_METRIC = "pr_auc"
BEST_SETTING_FOR_XAI = "pos1_bg2"

XAI_BG_SAMPLE = 20_000
XAI_TOP_TARGETS = 500



def load_feature_cols(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def recall_at_k_percent(y_true, y_score, k_percent):
    pos = y_true.sum()
    if pos == 0:
        return np.nan
    n = len(y_true)
    k = max(1, int(np.ceil(n * (k_percent / 100.0))))
    idx = np.argpartition(y_score, -k)[-k:]
    return float(y_true[idx].sum() / pos)


def make_block_split(df, val_frac, block_size_m):
    bx = (df["cell_x"].values // block_size_m).astype(np.int64)
    by = (df["cell_y"].values // block_size_m).astype(np.int64)
    h = (bx * 73856093) ^ (by * 19349663)
    h = np.abs(h) % 100
    return h < int(round(val_frac * 100))


def make_random_split(n, val_frac, seed):
    rng = np.random.default_rng(seed)
    return rng.random(n) < val_frac


def ensure_columns(df):
    if "label" not in df.columns or "has_any_ozmin_5km" not in df.columns:
        raise ValueError("Missing required columns")

    if "y" not in df.columns:
        df["y"] = (df["label"] > 0).astype(int)

    if "sample_weight" not in df.columns:
        df["sample_weight"] = 1.0
        pos = df["y"] == 1
        df.loc[pos, "sample_weight"] = df.loc[pos, "label"].astype(float)

    return df



def main():
    feature_cols = load_feature_cols(FEATURE_COLS_TXT)

    print("Loading data...")
    df = pd.read_parquet(DATA_TABLE)
    df = ensure_columns(df)

    feature_cols = [
        c for c in feature_cols
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not feature_cols:
        raise RuntimeError("No valid numeric features")

    y_all = df["y"].values.astype(int)
    pos_mask = y_all == 1
    bg_pool_mask = (y_all == 0) & (df["has_any_ozmin_5km"].values == 1)

    print(f"Rows: {len(df):,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Positives: {pos_mask.sum():,}")
    print(f"BG pool: {bg_pool_mask.sum():,}")


    use_spatial = (
        "cell_x" in df.columns
        and "cell_y" in df.columns
        and df["cell_x"].notna().any()
    )

    if use_spatial:
        print("Using spatial BLOCK split")
        is_val = make_block_split(df, VAL_FRAC, BLOCK_SIZE_M)
    else:
        print("Using RANDOM split")
        is_val = make_random_split(len(df), VAL_FRAC, RANDOM_SEED_SPLIT)

    is_train = ~is_val

    pos_val_idx = np.where(is_val & pos_mask)[0]
    bg_val_pool_idx = np.where(is_val & bg_pool_mask)[0]

    rng_val = np.random.default_rng(2025)
    n_bg_val = min(len(bg_val_pool_idx), max(10 * len(pos_val_idx), 50_000))
    bg_val_idx = (
        rng_val.choice(bg_val_pool_idx, size=n_bg_val, replace=False)
        if n_bg_val > 0 else np.array([], dtype=int)
    )

    val_idx = np.concatenate([pos_val_idx, bg_val_idx])
    y_val = y_all[val_idx]

    np.save(OUT_DIR / "val_idx.npy", val_idx)
    np.save(OUT_DIR / "y_val.npy", y_val)

    X_val = df.loc[val_idx, feature_cols].to_numpy(np.float32)

    pos_train_idx = np.where(is_train & pos_mask)[0]
    bg_train_pool_idx = np.where(is_train & bg_pool_mask)[0]

    N = len(df)
    metrics_rows = []


    for setting, cfg in SETTINGS.items():
        print(f"\n=== {setting} ===")

        setting_dir = OUT_DIR / setting
        pred_dir = setting_dir / "pred_full"
        val_dir = setting_dir / "val_probs"
        model_dir = setting_dir / "models"

        for d in [pred_dir, val_dir, model_dir]:
            d.mkdir(parents=True, exist_ok=True)

        best_score = -np.inf
        best_repeat = None
        best_model_path = None

        for r in range(cfg["repeats"]):
            seed = 1000 + r
            rng = np.random.default_rng(seed)

            if cfg["bg_mode"] == "mult":
                n_bg = cfg["k_bg"] * len(pos_train_idx)
                if n_bg > len(bg_train_pool_idx):
                    chosen_bg = bg_train_pool_idx
                else:
                    chosen_bg = rng.choice(bg_train_pool_idx, n_bg, replace=False)
            else:
                chosen_bg = bg_train_pool_idx

            train_idx = np.concatenate([pos_train_idx, chosen_bg])

            X_train = df.loc[train_idx, feature_cols].to_numpy(np.float32)
            y_train = y_all[train_idx]
            w_train = df.loc[train_idx, "sample_weight"].values.astype(np.float32)

            model = XGBClassifier(
                n_estimators=600,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,
                reg_lambda=1.0,
                reg_alpha=0.0,

                objective="binary:logistic",
                eval_metric="aucpr",

                tree_method="hist",   
                device="cuda",        

                random_state=seed,
            )


            print(f"Training repeat {r+1}/{cfg['repeats']} | n={len(train_idx):,}")
            model.fit(X_train, y_train, sample_weight=w_train)

            if SAVE_MODELS:
                path = model_dir / f"model_repeat_{r:02d}.joblib"
                joblib.dump(model, path)

            val_prob = model.predict_proba(X_val)[:, 1]
            pr_auc = average_precision_score(y_val, val_prob)
            r1 = recall_at_k_percent(y_val, val_prob, 1.0)
            r5 = recall_at_k_percent(y_val, val_prob, 5.0)

            score_now = {"pr_auc": pr_auc, "recall_at_1pct": r1, "recall_at_5pct": r5}[BEST_METRIC]

            if score_now > best_score:
                best_score = score_now
                best_repeat = r
                best_model_path = path

            np.save(val_dir / f"val_prob_repeat_{r:02d}.npy", val_prob.astype(np.float32))

            pred_path = pred_dir / f"pred_full_repeat_{r:02d}.npy"
            pred = np.memmap(pred_path, dtype="float32", mode="w+", shape=(N,))

            for s in range(0, N, PRED_CHUNK):
                e = min(N, s + PRED_CHUNK)
                Xc = df.iloc[s:e][feature_cols].to_numpy(np.float32)
                pred[s:e] = model.predict_proba(Xc)[:, 1]

            pred.flush()
            del pred

            metrics_rows.append({
                "setting": setting,
                "repeat": r,
                "pr_auc": pr_auc,
                "recall_at_1pct": r1,
                "recall_at_5pct": r5,
            })

        if best_model_path:
            shutil.copy2(best_model_path, setting_dir / "best_model.joblib")
            json.dump(
                {"best_metric": BEST_METRIC, "best_score": best_score, "best_repeat": best_repeat},
                open(setting_dir / "best_model_meta.json", "w"),
                indent=2
            )

        # Ensemble mean / std
        preds = sorted(pred_dir.glob("pred_full_repeat_*.npy"))
        arr = np.vstack([np.memmap(p, dtype="float32", mode="r") for p in preds])
        mean = arr.mean(axis=0).astype(np.float32)
        std = arr.std(axis=0).astype(np.float32)

        np.save(setting_dir / "pred_mean.npy", mean)
        np.save(setting_dir / "pred_std.npy", std)

    pd.DataFrame(metrics_rows).to_csv(OUT_DIR / "metrics_all_repeats.csv", index=False)
    print("\nDONE – XGBoost GPU ensemble completed.")


if __name__ == "__main__":
    main()
