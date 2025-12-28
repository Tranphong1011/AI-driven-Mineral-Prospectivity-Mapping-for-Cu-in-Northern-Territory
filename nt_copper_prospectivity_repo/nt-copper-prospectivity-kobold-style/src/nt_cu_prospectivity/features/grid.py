from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

@dataclass
class Grid:
    gdf: gpd.GeoDataFrame
    centroids: gpd.GeoDataFrame

def make_grid_from_aoi(aoi: gpd.GeoDataFrame, cell_size: float, crs_epsg: int) -> Grid:
    aoi = aoi.to_crs(epsg=crs_epsg)
    xmin, ymin, xmax, ymax = aoi.total_bounds
    xs = np.arange(xmin, xmax, cell_size)
    ys = np.arange(ymin, ymax, cell_size)

    polys = []
    ids = []
    k = 0
    for x in xs:
        for y in ys:
            polys.append(Polygon([(x,y),(x+cell_size,y),(x+cell_size,y+cell_size),(x,y+cell_size)]))
            ids.append(k); k += 1

    gdf = gpd.GeoDataFrame({"cell_id": ids}, geometry=polys, crs=f"EPSG:{crs_epsg}")
    gdf = gpd.overlay(gdf, aoi[["geometry"]], how="intersection")
    cent = gdf.copy()
    cent["geometry"] = cent.geometry.centroid
    return Grid(gdf=gdf, centroids=cent)
