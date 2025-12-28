from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from .metrics import pr_auc, recall_at_k_percent
from ..utils.io import ensure_dir, save_json

@dataclass
class PUResult:
    metrics: pd.DataFrame
    pred_mean: np.ndarray
    pred_std: np.ndarray

def _make_estimator(params: dict) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**params)

def train_pu_ensemble(X: np.ndarray, y: np.ndarray, settings: list[dict], estimator_params: dict, out_dir: Path, seed: int = 42) -> PUResult:
    rng = np.random.default_rng(seed)
    out_dir = ensure_dir(out_dir)
    pos_idx = np.where(y == 1)[0]
    bg_pool = np.where(y == 0)[0]

    all_preds = []
    rows = []

    for s in settings:
        name = s["name"]; bg_ratio = s["bg_ratio"]; repeats = int(s["repeats"])
        for r in range(repeats):
            if bg_ratio == "all":
                bg_idx = bg_pool
            else:
                n_bg = int(len(pos_idx) * int(bg_ratio))
                bg_idx = rng.choice(bg_pool, size=min(n_bg, len(bg_pool)), replace=False)

            train_idx = np.concatenate([pos_idx, bg_idx])
            rng.shuffle(train_idx)

            est = _make_estimator(estimator_params)
            est.fit(X[train_idx], y[train_idx])

            pred = est.predict_proba(X)[:, 1]
            all_preds.append(pred)

            rows.append({
                "setting": name,
                "repeat": r,
                "pr_auc": pr_auc(y, pred),
                "recall@1%": recall_at_k_percent(y, pred, 1.0),
                "recall@5%": recall_at_k_percent(y, pred, 5.0),
                "n_pos": int(len(pos_idx)),
                "n_bg": int(len(bg_idx)),
            })

            joblib.dump(est, out_dir / f"model_{name}_r{r:02d}.joblib")

    preds = np.vstack(all_preds)
    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)

    metrics_df = pd.DataFrame(rows).sort_values(["setting","repeat"])
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    save_json(out_dir / "summary.json", {"n_models": int(preds.shape[0]), "seed": seed, "estimator_params": estimator_params})
    return PUResult(metrics=metrics_df, pred_mean=pred_mean, pred_std=pred_std)
