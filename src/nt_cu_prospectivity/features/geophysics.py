from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from ..utils.raster import sample_raster_at_points

def build_geophysics_features(mag_raster: Path, xs: np.ndarray, ys: np.ndarray, aem_raster: Path | None = None) -> pd.DataFrame:
    mag = sample_raster_at_points(mag_raster, xs, ys)
    out = {"mag": mag}
    if aem_raster:
        out["aem"] = sample_raster_at_points(aem_raster, xs, ys)
    return pd.DataFrame(out)
