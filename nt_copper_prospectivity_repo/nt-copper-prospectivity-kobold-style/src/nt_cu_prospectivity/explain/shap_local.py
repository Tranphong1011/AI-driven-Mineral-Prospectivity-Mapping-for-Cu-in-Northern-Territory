from __future__ import annotations
from pathlib import Path
import pandas as pd
import shap
import joblib
from ..utils.io import ensure_dir

def run_shap_local(model_path: Path, X: pd.DataFrame, row_ids: list[int], out_dir: Path) -> list[Path]:
    out_dir = ensure_dir(out_dir)
    model = joblib.load(model_path)
    explainer = shap.Explainer(model, X)
    outs = []
    import matplotlib.pyplot as plt
    for rid in row_ids:
        sv = explainer(X.iloc[[rid]])
        fig_path = out_dir / f"shap_local_row{rid}.png"
        shap.plots.waterfall(sv[0], show=False)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        outs.append(fig_path)
    return outs
