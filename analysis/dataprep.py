import geopandas as gpd
import pandas as pd
import numpy as np
import rioxarray

from glob import glob
from pathlib import Path
from tqdm import tqdm
from geocube.api.core import make_geocube

from analysis import prepare

def remove_intersecting_crowns(larger_geo_df, smaller_geo_df):
    return larger_geo_df.loc[~larger_geo_df.intersects(smaller_geo_df.unary_union)].reset_index(drop=True)

def get_geo_df(polys, epsg_code:str):
    coords_gdf = gpd.GeoDataFrame(crs=epsg_code, geometry=polys)
    return coords_gdf

def extract_single_polygons(aoi_filepath, epsg_code):
    aois = gpd.read_file(aoi_filepath)
    index = 0
    out_dir = Path(aoi_filepath).parent
    for aoi in aois['geometry']:
        index = index + 1
        gdf = get_geo_df(aoi, epsg_code)
        outfile_name = str(index) + '_aoi.gpkg'
        gdf.to_file(out_dir + outfile_name, driver="GPKG")
    print("aoi files destinations", out_dir)

def remap_tree_species(geo_df):
    species_map = {
        'Pinus sylvestris (abgängig)': 'deadwood',
        'Pinus sylvestris (tot)': 'deadwood',
        'Picea abies (abgängig)': 'deadwood',
        'Picea abies (tot)': 'deadwood',
        'Fagus sylvatica (abgängig)': 'deadwood',
        'agus sylvatica': 'Fagus sylvatica',
        'Fagus sylvatica (tot)': 'deadwood',
        'Larix (abgängig)': 'deadwood',
        'Quercus': 'Quercus spec.',
        'Larix': 'Larix decidua',
        'Pseudotsuga mentiesii': 'Pseudotsuga menziesii',
        'Quercus (abgängig)': 'deadwood',
        'Acer': 'Acer pseudoplatanus',
        'Lbh (abgängig)': 'deadwood'
    }
    species_id_map = {
        'deadwood': 8,
        'Abies alba': 10,
        'Larix decidua': 11,
        'Picea abies': 12,
        'Pinus sylvestris': 13,
        'Pseudotsuga menziesii': 14,
        'forest floor': 9,
        'Acer pseudoplatanus': 2,
        'Fraxinus excelsior': 5,
        'Fagus sylvatica': 4,
        'Quercus spec.': 6,
        'Tilia': 16,
        'Lbh': 18,
        'Sorbus torminalis': 19,
        'Ulmus': 20,
        'Acer platanoides': 21,
        'Quercus rubra': 22
    }
    remapped_shp_file = geo_df.copy()
    remapped_shp_file['species'] = remapped_shp_file['species'].map(species_map)
    remapped_shp_file['species_ID'] = remapped_shp_file['species'].map(species_id_map)
    return remapped_shp_file

def clean_entries(geo_df):
    searchfor = ['?', '? (abgängig)', 'agus sylvatica']
    return geo_df[~geo_df.isin(searchfor).any(axis=1)]

def clip_poly_from_aoi(aois_dir, gt_crowns_filepath, name_base:str, epsg:int):
    pred_crowns = gpd.read_file(gt_crowns_filepath)
    aoi_files = sorted(glob(aois_dir + '/*.gpkg'))
    out_dir = Path(gt_crowns_filepath) / 'pred_crown_tiles'
    out_dir.mkdir(parents=True, exist_ok=True)
    index = 0
    for aoi_file in aoi_files:
        aoi = gpd.read_file(aoi_file)
        geom = aoi['geometry'][0]
        name = str(out_dir) + '/' + str(index) + name_base + '.gpkg'
        poly_container = []
        scores = []
        for poly, score in zip(pred_crowns['geometry'], pred_crowns['Confidence_score']):
            if geom.contains(poly):
                poly_container.append(poly)
                scores.append(score)
        new_df = get_geo_df(poly_container, epsg)
        new_df['Confidence_score'] = scores
        new_df.to_file(name, driver="GPKG")
        index = index + 1
    print("shape file folder", out_dir)

def clip_poly_and_keep_labels_from_aoi_(aois_dir, gt_crowns_filepath, name_base:str, epsg:int):
    labels = gpd.read_file(gt_crowns_filepath)
    aoi_files = sorted(glob(aois_dir + '/*.gpkg'))
    out_dir = Path(gt_crowns_filepath).parent
    index = 0
    for aoi_file in aoi_files:
        aoi = gpd.read_file(aoi_file)
        geom = aoi['geometry'][0]
        name = str(out_dir) + str(index) + name_base
        poly_container = []
        species = []
        for poly, instance in zip(labels['geometry'], labels['Art']):
            if not geom.contains(poly):
                poly_container.append(poly)
                species.append(instance)
        new_df = get_geo_df(poly_container, epsg)
        new_df['species'] = species
        new_df['species_ID'] = 1
        
        #TODO only rempa and clean entries on adjusted gt file, species mapping via mask file
        new_df = remap_tree_species(new_df)
        new_df = clean_entries(new_df)

        new_df.to_file(name, driver="GPKG")
        index = index + 1
    print("shape file destination", out_dir)

def create_aoi_ortho_tiles(aois_dir, ortho_filepath, name_base:str):
    aoi_files = glob.glob(str(aois_dir) + '/*.gpkg')
    output_path = Path(ortho_filepath).parent
    index = 0
    for aoi in aoi_files:
        try:
            out_name = output_path + str(index) + name_base
            prepare.create_mask_from_shapefile(aoi, ortho_filepath, out_name)
            index = index + 1
        except ValueError:
            index = index + 1
            continue

def extract_polys_from_aois(aois_dir, pred_crown_filepath, name_base:str, epsg:int, reverse:bool=False):
    aois = sorted(glob(aois_dir + '*.gpkg'))
    pred_crowns = gpd.read_file(pred_crown_filepath)
    index = 0
    out_dir = Path(pred_crown_filepath).parent / 'pred_crown_tiles_gt_aois/'
    for aoi_file in aois:
        aoi = gpd.read_file(aoi_file)
        name = str(out_dir) + str(index) + name_base
        poly_container = []
        scores = []
        for poly, score in zip(pred_crowns['geometry'], pred_crowns['Confidence_score']):
            if reverse == False:
                if aoi['geometry'][0].contains(poly):
                    poly_container.append(poly)
                    scores.append(score)
            if reverse == True:
                if not aoi['geometry'][0].contains(poly):
                    poly_container.append(poly)
                    scores.append(score)
                # TODO else implement if it intersects
        new_df = get_geo_df(poly_container, epsg)
        new_df['Confidence_score'] = scores
        new_df.to_file(name, driver="GPKG")
        index = index + 1

def remove_polygons_inside_aoi():
    #TODO übertrage funktion
    return

def make_image_mask(gt_crown_filepath, ortho_filepath):
    source_ds = gpd.read_file(gt_crown_filepath)
    dataset = rioxarray.open_rasterio(ortho_filepath, masked=True)
    Cube = make_geocube(vector_data=source_ds, like=dataset)
    out_dir = Path(gt_crown_filepath).parent / 'gt_masks'
    out_file = str(out_dir) + str(Path(ortho_filepath).stem) + '_mask.tif'
    Cube['species_ID'].rio.to_raster(out_file)

def make_multiple_image_masks(gt_crowns_dir, orthos_dir):
    gt_crown_files = sorted(glob.glob(gt_crowns_dir + '*.gpkg'))
    ortho_files = sorted(glob.glob(orthos_dir + '*.tif'))
    for gt_crown, ortho in zip(gt_crown_files, ortho_files):
        make_image_mask(gt_crown, ortho)

def get_species_distribution(crowns_dir, mode:str):
    gt_crown_files = sorted(glob(crowns_dir + '*.gpkg'))
    if len(gt_crown_files) == 0:
        gt_crown_files = sorted(glob(crowns_dir + 'poly*.shp'))
    dfs = [gpd.read_file(gt_file) for gt_file in gt_crown_files]
    concat_df = pd.concat(dfs)
    if mode == 'percent':
        return concat_df['species'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
    elif mode == 'decimal':
        return concat_df.value_counts(['species_ID', 'species'])

def get_str_num(name, pos):
    short_name = Path(name).stem
    return short_name.split('_')[pos]

def get_image_species_distribution(single_tree_dir):
    f = sorted(glob(single_tree_dir + '*.png'))
    numbers = [get_str_num(str(filename), 6) for filename in f]
    return sorted(set(numbers))

