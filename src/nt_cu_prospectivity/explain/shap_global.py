from __future__ import annotations
from pathlib import Path
import pandas as pd
import shap
import joblib
from ..utils.io import ensure_dir

def run_shap_global(model_path: Path, X: pd.DataFrame, out_dir: Path, max_samples: int = 5000) -> Path:
    out_dir = ensure_dir(out_dir)
    model = joblib.load(model_path)
    Xs = X.sample(n=max_samples, random_state=42) if len(X) > max_samples else X
    explainer = shap.Explainer(model, Xs)
    sv = explainer(Xs)
    import matplotlib.pyplot as plt
    fig_path = out_dir / "shap_summary.png"
    shap.plots.beeswarm(sv, show=False)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    return fig_path
