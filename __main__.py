from analysis import label_tools
from analysis import prepare
from analysis import visualization
from analysis import preprocessing
from analysis import features

from pathlib import Path

def main():
    # define dataset paths folders, Schiefer or Stadtwald, probably multiple files for one site
    site_folder = '/home/richard/data/Schiefer/'
    orthophoto_dir = site_folder + 'orthophotos'
    gt_mask_dir = site_folder + 'gt_masks'
    prediction_dir = site_folder + 'predictions'
    # gt_annotations_dir = "" # optional
    
    assert Path(site_folder).exists()
    assert Path(orthophoto_dir).exists()
    assert Path(gt_mask_dir).exists()
    assert Path(prediction_dir).exists()

    # 1. extract single tree crowns (polygons) from orthophoto
    if not Path(site_folder + 'pred_polygon_clipped_raster_files').exists():
        prepare.clip_crown_sets_with_gt_masks(orthophoto_dir, prediction_dir, gt_mask_dir, make_squares=False, step_size=0.5)

    # 2. preprocess images create two sets
    first_set = preprocessing.preprocess_images(site_folder + 'pred_polygon_clipped_raster_files', None, None, False, True, True, False)
    second_set = preprocessing.preprocess_images(site_folder + 'pred_polygon_clipped_raster_files', None, None, False, True, False, False)

    # # debug
    # first_set = '/home/richard/data/Schiefer/preprocessed_None_clahe-denoising_clipped_pred_polygon_204'
    # second_set = '/home/richard/data/Schiefer/preprocessed_None_clahe_clipped_pred_polygon_204'

    # 3. feature extraction
    cnns = ['vgg', 'resnet', 'effnet']
    le_and_encoding_paths = []
    for cnn in cnns:
        clahe_denoising = features.create_and_save_le_encodings(cnn, first_set, site_folder)
        clahe = features.create_and_save_le_encodings(cnn, second_set, site_folder)
        le_and_encoding_paths.append(clahe_denoising)
        le_and_encoding_paths.append(clahe)
    print(le_and_encoding_paths)
    
    # 4. analysis


if __name__ == "__main__":
    main()