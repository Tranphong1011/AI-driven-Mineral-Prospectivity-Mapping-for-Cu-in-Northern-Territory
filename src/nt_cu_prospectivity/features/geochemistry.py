from __future__ import annotations
from pathlib import Path
import pandas as pd
import geopandas as gpd

def build_geochem_features(geochem_file: Path, grid_centroids: gpd.GeoDataFrame, crs_epsg: int) -> pd.DataFrame:
    # TODO: migrate full aggregation/anomaly logic from your notebook.
    gdf = gpd.read_file(geochem_file).to_crs(epsg=crs_epsg)
    joined = gpd.sjoin_nearest(grid_centroids, gdf[["geometry"]], how="left", distance_col="dist_geochem_m")
    return pd.DataFrame({"dist_geochem_m": joined["dist_geochem_m"].values}, index=grid_centroids.index)
