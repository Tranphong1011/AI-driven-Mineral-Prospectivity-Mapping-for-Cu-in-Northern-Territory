from __future__ import annotations
from pathlib import Path
import pandas as pd
import geopandas as gpd
from ..utils.vector import read_vector

def build_geology_features(geology_file: Path, geology_layer: str, grid_polys: gpd.GeoDataFrame, crs_epsg: int) -> pd.DataFrame:
    geo = read_vector(geology_file, layer=geology_layer, to_crs_epsg=crs_epsg)
    cent = grid_polys.copy()
    cent["geometry"] = cent.geometry.centroid
    sj = gpd.sjoin(cent, geo[["geometry"]], how="left", predicate="intersects")
    return pd.DataFrame({"geo_hit": (~sj.index_right.isna()).astype(int).values}, index=grid_polys.index)
