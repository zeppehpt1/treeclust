from analysis import label_tools
from analysis import prepare
from analysis import visualization
from analysis import preprocessing

from pathlib import Path

# define dataset paths folders, Schiefer or Stadtwald, probably multiple files for one site
site_folder = ""
orthophoto_dir = ""
prediction_dir = ""

gt_mask_dir = ""
# gt_annotations_dir = "" # optional

# 1. extract single tree crowns (polygons) from orthophoto
if not Path(site_folder / 'pred_polygon_clipped_raster_files').exists():
    prepare.clip_crown_sets_with_gt_masks(orthophoto_dir, prediction_dir, gt_mask_dir, make_squares=False, step_size=0.5)

# 2. preprocess images create two sets
first_set = preprocessing.preprocess_images(site_folder / 'pred_polygon_clipped_raster_files', None, None, False, True, True, False)
second_set = preprocessing.preprocess_images(site_folder / 'pred_polygon_clipped_raster_files', None, None, False, True, False, False)

# 3. feature extraction
