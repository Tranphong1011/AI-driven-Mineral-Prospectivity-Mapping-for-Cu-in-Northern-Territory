from __future__ import annotations
from pathlib import Path
import numpy as np
import joblib

def load_models(model_dir: Path) -> list[Path]:
    return sorted(model_dir.glob("model_*.joblib"))

def predict_ensemble(model_paths: list[Path], X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    preds = []
    for p in model_paths:
        m = joblib.load(p)
        preds.append(m.predict_proba(X)[:, 1])
    arr = np.vstack(preds)
    return arr.mean(axis=0), arr.std(axis=0)
