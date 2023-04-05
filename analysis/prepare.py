import geopandas as gpd
from geopandas import GeoDataFrame
import polygonize
import shapely.wkt
import rasterio
import json
import numpy as np
import os
import glob
from rasterio.mask import mask
from shapely import geometry
from math import sin, cos, radians
from pathlib import Path
from typing import Union
from tqdm import tqdm

# TODO: actually next step
# TODO: Remove original crown file from other repo
# TODO: clean up functions and align them to coding convention PEP

def highest_pixel_count(img_array):
    pixel, n_of_pixels = np.unique(img_array, return_counts=True)
    highest_pixel_value = pixel[np.argsort(-n_of_pixels)]
    if highest_pixel_value[0] == 0 and len(highest_pixel_value) == 1:
        return 9999
    index = 0
    forbidden_values = {0,1,2,9} # discard forest floor as well
    while True:
        if highest_pixel_value[index] not in forbidden_values:
            return highest_pixel_value[index]
        elif 0 <= index < len(highest_pixel_value):
            return 9999
        index += 1

def remove_images_outside_of_gt_area(files):
    for path in files:
        number = path.stem.split('_')[5]
        if number == 9999:
            os.remove(path)

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

def remove_none_rows(gdf):
    gdf = gdf.replace(to_replace='None', value=np.nan).dropna()
    gdf = gdf.reset_index(drop=True)
    return gdf

def clip_crown_from_raster(img_path:str, ortho_mask_tif:str, polygon,out_file_suffix:str, make_squares:bool):
    # includes gt mask data
    # TODO: create a flag to switch between gt and no gt
    # TODO: Adjust Path
    # TODO: remove or add extra functionality to include gt or not, in dev right now
    # get ground truth of ortho mask (labeled schiefer)
    ortho_mask_data = rasterio.open(ortho_mask_tif)
    epsg = int(str(ortho_mask_data.crs).split(':')[1])
    # rasterio wants coordinates as a geodata geojson format for parsing
    geo = gpd.GeoDataFrame({'geometry':polygon},index=[0],crs=epsg)
    geo = geo.to_crs(epsg=epsg)
    coords = get_geo_features(geo)
    # define output params
    ortho_mask_out, out_transform = mask(ortho_mask_data,shapes=coords,crop=True)
    out_meta = ortho_mask_data.meta.copy()
    out_meta.update({
        "driver":"PNG",
        "height": ortho_mask_out.shape[1],
        "width": ortho_mask_out.shape[2],
        "transform": out_transform,
        "crs":epsg
    })
    #highest pixel count (excluding black pixels) of ground truth area
    data_array = ortho_mask_out
    highest_pixel_value = highest_pixel_count(data_array) # ensures to discard black and white majority of pixels
    # check for polygons outside of ground truth area and invalid ones (black, white and forest floor pixels are discarded)
    if highest_pixel_value == 9999:
        return
    
    # get original polygon mask
    data = rasterio.open(img_path)
    epsg = int(str(data.crs).split(':')[1])
    # rasterio wants coordinates as a geodata geojson format for parsing
    geo = gpd.GeoDataFrame({'geometry':polygon},index=[0],crs=epsg)
    geo = geo.to_crs(epsg=epsg)
    coords = get_geo_features(geo)
    # define output params
    out_img, out_transform = mask(data,shapes=coords,crop=True)
    out_meta = data.meta.copy()
    out_meta.update({
        "driver":"PNG",
        "height": out_img.shape[1],
        "width": out_img.shape[2],
        "transform": out_transform,
        "crs":epsg
    })
    
    shape = 'polygon'
    if make_squares == True:
        shape = 'square'
    
    # saving poly clip and attatch highest pixel count value to filename
    out_dir = Path(img_path).parent.parent / Path('pred_' + shape + '_clipped_raster_files')
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    output_path = out_dir / Path(str(Path(img_path).stem) + out_file_suffix + str(highest_pixel_value) + '.png')
    with rasterio.open(output_path,"w",**out_meta) as dest:
        dest.write(out_img)

def clip_multiple_crowns_from_raster(img_path: Union[str,Path],crowns:GeoDataFrame, make_squares, step_size):
    """Create multiple png files of tree crowns based on polygons. Clips either the whole tree crown or inner square of the polygon.

    Args:
        img_path (Union[str,Path]): Path to image
        crowns (GeoDataFrame): 'geometry' column with polygons
        make_squares (bool, optional): Defaults to False. Set to True if you want to get the most inner square of the polygons.
        step_size: Defaults to 0.5. Affects only inner square polygons.
    """
    ortho_mask_path = '/home/richard/bakim/data/Schiefer/roh/masks/CFB184_ortho_mask_byte.tif'
    assert Path(ortho_mask_path).exists()
    
    if make_squares == True:
        crowns = get_gdf_with_inner_square_polygons(crowns,step_size)
    for index in (pbar := tqdm(range(len(crowns['geometry'])), leave=False)):
        pbar.set_description(f"Processing number {index}")
        file_suffix = '_mask_{0:0>4}_'.format(index)
        clip_crown_from_raster(img_path, ortho_mask_path, crowns['geometry'][index],file_suffix)
        
def clip_crowns_with_gt_mask(img_path: Union[str, Path], crowns: GeoDataFrame, mask_path: Union[str, Path], make_squares, step_size):
    """Create multiple png files of tree crowns based on polygons. Clips either the whole tree crown or inner square of the polygon. Includes the ground truth value of the provided mask image for analysis.

    Args:
        img_path (Union[str,Path]): Path to image
        crowns (GeoDataFrame): 'geometry' column with polygons
        mask_path (Union[str,Path]): Path to mask image
        make_squares (bool, optional): Defaults to False. Set to True if you want to get the most inner square of the polygons.
        step_size: Defaults to 0.5. Affects only inner square polygons.
    """
    
    if make_squares == True:
        crowns = get_gdf_with_inner_square_polygons(crowns,step_size)
    for index in (pbar := tqdm(range(len(crowns['geometry'])), leave=False)):
        pbar.set_description(f"Processing number {index}")
        file_suffix = '_mask_{0:0>4}_'.format(index)
        clip_crown_from_raster(img_path, mask_path, crowns['geometry'][index],file_suffix)

def clip_crown_sets_with_gt_masks(imgs_dir: Union[str, Path], crown_dir, mask_dir: Union[str, Path], make_squares, step_size):
    """Creates multiple png file sets of tree crowns based on polygons. Clips either the whole tree crown or inner square of the polygon. Includes the ground truth value of the provided mask image for analysis.

    Args:
        img_path (Union[str,Path]): Path to image
        crowns (GeoDataFrame): 'geometry' column with polygons
        mask_path (Union[str,Path]): Path to mask image
        make_squares (bool, optional): Defaults to False. Set to True if you want to get the most inner square of the polygons.
        step_size: Defaults to 0.5. Affects only inner square polygons when make_squares is set to TRUE.
    """
    
    img_files = glob.glob(imgs_dir + '/*.tif')
    crown_files = glob.glob(crown_dir + '/*.gpkg')
    mask_files = glob.glob(mask_dir + '/*.tif')
    
    [folder.sort() for folder in [img_files, crown_files, mask_files]]
    
    assert len(img_files) == len(crown_files) == len(mask_files)
    
    make_squares = make_squares
    step_size = step_size
    for img, crown_file, mask in zip(img_files, crown_files, mask_files):
        crowns = gpd.read_file(crown_file)
        clip_crowns_with_gt_mask(img, crowns, mask, make_squares, step_size)
    
    # combine clipped imgs into one folder
    # call comb function
    # already combines all into one folder...

def combine_clipped_crowns(clipped_crowns_dir):
    
    clipped_crown_sets = glob.glob(clipped_crowns_dir + '/CFB*')
    all_clipped_crowns_dir = clipped_crowns_dir.parent / 'all_crowns'
    Path(all_clipped_crowns_dir).mkdir(parents=True, exist_ok=True)
    print(all_clipped_crowns_dir)
    
    clipped_crown_files = [glob.glob(clipped_set + '/*.png') for clipped_set in clipped_crown_sets]
    print(len(clipped_crown_files))

    
def get_gdf_with_inner_square_polygons(crowns:GeoDataFrame,step_size=1) -> GeoDataFrame:
    crowns['rec_poly'] = [get_inner_square_corner_coordinates_from_polygon(step_size,polygon) for polygon in crowns['geometry']]
    crowns = remove_none_rows(crowns) # very small squares
    crowns['rec_poly'] = [points_to_polygon(rec_points) for rec_points in crowns['rec_poly']]
    crowns = crowns.drop(columns='geometry')
    crowns = crowns.rename(columns={"rec_poly":"geometry"})
    return crowns

if __name__ == "__main__":
    print("write a test case or something")