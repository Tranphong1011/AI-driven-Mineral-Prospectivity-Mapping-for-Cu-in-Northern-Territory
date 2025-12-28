from __future__ import annotations
import geopandas as gpd

def make_positive_buffers(gdf: gpd.GeoDataFrame, buffer_m: float) -> gpd.GeoDataFrame:
    out = gdf.copy(); out['geometry']=out.geometry.buffer(buffer_m); return out
