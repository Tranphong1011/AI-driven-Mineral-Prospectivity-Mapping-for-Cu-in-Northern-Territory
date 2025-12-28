import geopandas as gpd
from shapely.geometry import box
from nt_cu_prospectivity.features.grid import make_grid_from_aoi

def test_grid():
    aoi=gpd.GeoDataFrame({'id':[0]}, geometry=[box(0,0,1000,1000)], crs='EPSG:28352')
    grid=make_grid_from_aoi(aoi,500,28352)
    assert len(grid.gdf)>0
