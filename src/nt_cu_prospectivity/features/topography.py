from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from ..utils.raster import sample_raster_at_points

def build_dem_features(dem_path: Path, xs: np.ndarray, ys: np.ndarray) -> pd.DataFrame:
    # TODO: migrate full DEM derivatives from your notebook.
    elev = sample_raster_at_points(dem_path, xs, ys)
    return pd.DataFrame({"dem_elev": elev})
