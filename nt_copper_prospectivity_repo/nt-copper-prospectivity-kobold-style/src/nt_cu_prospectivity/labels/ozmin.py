from __future__ import annotations
from pathlib import Path
import geopandas as gpd

def load_ozmin(path: Path, layer: str, crs_epsg: int) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path, layer=layer)
    if gdf.crs is None:
        raise ValueError("OZMIN has no CRS.")
    return gdf.to_crs(epsg=crs_epsg)
