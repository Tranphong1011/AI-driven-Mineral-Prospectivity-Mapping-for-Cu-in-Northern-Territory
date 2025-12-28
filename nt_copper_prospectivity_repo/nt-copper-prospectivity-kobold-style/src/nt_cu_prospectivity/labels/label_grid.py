from __future__ import annotations
import geopandas as gpd
import pandas as pd

def label_grid_by_buffers(grid_polys: gpd.GeoDataFrame, buffers: gpd.GeoDataFrame) -> pd.Series:
    joined = gpd.sjoin(grid_polys[["geometry"]], buffers[["geometry"]], how="left", predicate="intersects")
    pos = (~joined.index_right.isna()).groupby(level=0).any()
    return pos.reindex(grid_polys.index, fill_value=False).astype(int)
