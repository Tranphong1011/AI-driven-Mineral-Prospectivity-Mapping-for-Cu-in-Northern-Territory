from __future__ import annotations
from pathlib import Path
import geopandas as gpd

def read_vector(path: Path, layer: str | None = None, to_crs_epsg: int | None = None) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, layer=layer)
    return gdf.to_crs(epsg=to_crs_epsg) if to_crs_epsg else gdf
