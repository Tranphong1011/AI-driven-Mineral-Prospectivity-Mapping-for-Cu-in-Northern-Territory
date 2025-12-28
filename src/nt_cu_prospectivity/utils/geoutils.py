from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box

@dataclass(frozen=True)
class AOI:
    gdf: gpd.GeoDataFrame

def load_aoi(aoi_file: Path | None, aoi_layer: str | None, bbox: list[float] | None, crs_epsg: int) -> AOI:
    if aoi_file:
        gdf = gpd.read_file(aoi_file, layer=aoi_layer)
        if gdf.crs is None:
            raise ValueError("AOI has no CRS.")
        return AOI(gdf.to_crs(epsg=crs_epsg))
    if bbox:
        gdf = gpd.GeoDataFrame({"id":[0]}, geometry=[box(*bbox)], crs=f"EPSG:{crs_epsg}")
        return AOI(gdf=gdf)
    raise ValueError("Provide either aoi_file or bbox.")
