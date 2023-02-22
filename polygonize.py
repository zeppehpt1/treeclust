import geopandas as gpd
import shapely.wkt
import rasterio
import json
from rasterio.mask import mask
from geopandas import GeoDataFrame
from shapely import geometry
from math import sin, cos, radians
from pathlib import Path

# TODO: actually next step
# TODO: Remove original crown file from other repo

def get_centroid(polygon):
    centroid_str = polygon.centroid.wkt
    return shapely.wkt.loads(centroid_str)

def point_inside_shape(point, polygon) -> bool:
    point = gpd.GeoDataFrame(geometry=[point])
    return(point.within(polygon).iloc[0])

def rotated_square(cx, cy, size=70, degrees=0):
    """ Calculate coordinates of a rotated square or normal one centered at 'cx, cy'
        given its 'size' and rotation by 'degrees' about its center.
    """
    h = size/2
    l, r, b, t = cx-h, cx+h, cy-h, cy+h
    a = radians(degrees)
    cosa, sina = cos(a), sin(a)
    pts = [(l, b), (l, t), (r, t), (r, b)]
    return [(( (x-cx)*cosa + (y-cy)*sina) + cx,
             (-(x-cx)*sina + (y-cy)*cosa) + cy) for x, y in pts]

def make_shapely_points(points):
    return [geometry.Point(point) for point in points]

def get_inner_square_corner_coordinates_from_polygon(step_size,polygon):
    
    result_points = []
    square_size = 70 # has enough buffer and suitable for all crowns
    centroid = get_centroid(polygon)
    square_coordinates = rotated_square(centroid.x,centroid.y,square_size,degrees=0) # initial square size
    while(square_size != 0):
        result_points = square_coordinates
        if all(point_inside_shape(geometry.Point(point),polygon) for point in square_coordinates):
            return result_points
        else: 
            # rectangle is outside of polygon
            square_coordinates = rotated_square(centroid.x,centroid.y,square_size,degrees=0)
            square_size -= step_size
            
def points_to_polygon(points):
    
    polygon_list = make_shapely_points(points)
    return geometry.Polygon([[p.x, p.y] for p in polygon_list])

def visualize_two_polygons(first_polygon,second_polygon):
    
    poly_gdf = gpd.GeoDataFrame({"id": [1,2], "geometry": [first_polygon,second_polygon]}, geometry="geometry")
    poly_gdf.plot(facecolor="None", edgecolor="red")
    
def get_geo_features(gdf:GeoDataFrame) -> json:
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

def clip_crown_from_raster(img_path:str,polygon,out_file_suffix:str):
    
    # TODO: Solve annoying crs init warning
    # TODO: Adjust Path
    
    out_dir = Path(img_path).parent.parent / Path(img_path.split('_')[0] + '_clipped_raster_files')
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    output_path = out_dir / Path(str(Path(img_path).stem) + out_file_suffix)
    
    data = rasterio.open(img_path)
    epsg = int(str(data.crs).split(':')[1])
    
    # rasterio wants coordinates as a geodata geojson format for parsing
    geo = gpd.GeoDataFrame({'geometry':polygon},index=[0],crs=epsg)
    geo = geo.to_crs(crs=data.crs.data)
    coords = get_geo_features(geo)
    
    out_img, out_transform = mask(data,shapes=coords,crop=True)
    out_meta = data.meta.copy()
    out_meta.update({
        "driver":"PNG",
        "height": out_img.shape[1],
        "width": out_img.shape[2],
        "transform": out_transform,
        "crs":epsg
    })
    # saving poly clip
    with rasterio.open(output_path,"w",**out_meta) as dest:
        dest.write(out_img)

def clip_multiple_crowns_from_raster(img_path,crowns:GeoDataFrame,make_squares=False):
    
    if make_squares == True:
        crowns = get_gdf_with_inner_square_polygons
    
    for index in range(len(crowns['geometry'])):
        file_suffix = '_{0:0>4}.png'.format(index)
        clip_crown_from_raster(img_path,crowns['geometry'][index],file_suffix)

def get_gdf_with_inner_square_polygons(crowns:GeoDataFrame,step_size=1) -> GeoDataFrame:
    
    crowns['rec_poly'] = [get_inner_square_corner_coordinates_from_polygon(step_size,polygon) for polygon in crowns['geometry']]
    crowns['rec_poly'] = [points_to_polygon(rec_points) for rec_points in crowns['rec_poly']]
    crowns = crowns.drop(columns='geometry')
    crowns = crowns.rename(columns={"rec_poly":"geometry"})
    return crowns
    
if __name__ == "__main__":
    print("write a test case or something")