from analysis import label_tools
from analysis import prepare
from analysis import visualization
from analysis import preprocessing
from analysis import features as ft
from analysis import cluster

from pathlib import Path
import csv
import pickle
import pandas as pd

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

    # # 1. extract single tree crowns (polygons) from orthophoto
    # if not Path(site_folder + 'pred_polygon_clipped_raster_files').exists():
    #     prepare.clip_crown_sets_with_gt_masks(orthophoto_dir, prediction_dir, gt_mask_dir, make_squares=False, step_size=0.5)
    # else:
    #     print('Cropped files already exists')

    # # 2. preprocess images create two sets
    # first_set = preprocessing.preprocess_images(site_folder + 'pred_polygon_clipped_raster_files', None, None, False, True, True, False)
    # second_set = preprocessing.preprocess_images(site_folder + 'pred_polygon_clipped_raster_files', None, None, False, True, False, False)

    # # debug
    # first_set = '/home/richard/data/Schiefer/preprocessed_None_clahe-denoising_clipped_pred_polygon_204'
    # second_set = '/home/richard/data/Schiefer/preprocessed_None_clahe_clipped_pred_polygon_204'

    # # 3. feature extraction
    # cnns = ['vgg', 'resnet', 'effnet']
    # le_and_encoding_paths = []
    # for cnn in cnns:
    #     clahe_denoising = ft.create_and_save_le_encodings(cnn, first_set, site_folder)
    #     clahe = ft.create_and_save_le_encodings(cnn, second_set, site_folder)
    #     le_and_encoding_paths.append(clahe_denoising)
    #     le_and_encoding_paths.append(clahe)
    # print(le_and_encoding_paths)
    # # TODO: implement a way to read feature files in folder --> list + sort them
    # now manually for debugging
    encoding_dir_path = Path('/home/richard/data/Schiefer/encodings/')
    le_dir_path = Path('/home/richard/data/Schiefer/label_encodings/')
    encoding_files = sorted(encoding_dir_path.glob('*.pickle'))
    le_files = sorted(le_dir_path.glob('*.pickle'))
    
    # define lists for pickle dump
    preprocessing_types = []
    cnns = []
    dr_techniques = []
    cluster_techniques = []
    micro_f1_scores = []
    macro_f1_scores = []
    nmi_scores = []
    mean_squared_erros = []
    
    # 4. analysis
    for encoding_file, le_file in zip(encoding_files, le_files):
        print("enfile", encoding_file)
        print("lefile", le_file)
        files, features, labels, y_gt = cluster.load_all_data(encoding_file, le_file)
        # dr
        pca_reduced = cluster.pca(features)
        umap_reduced = cluster.umap(features)
        reduced_features = [pca_reduced, umap_reduced] # TODO separate function
        dr_idents = ['pca', 'umap']
        # name
        used_preprocess = str(Path(encoding_file).stem).split('_')[1]
        used_cnn = str(Path(encoding_file).stem).split('_')[0]
        
        for dr_features, dr_ident in zip(reduced_features, dr_idents):
            if dr_ident == 'pca':
                dr_ident = 'pca'
            elif dr_ident == 'umap':
                dr_ident = 'umap'
            used_dr = dr_ident
            
            used_cluster_alg, aquired_micro_f1_scores, aquired_macro_f1_scores, aquired_nmi_scores = cluster.run_cluster(dr_features, labels, y_gt)
            for cl_alg, micro_f1_score, macro_f1_score, nmi_score in zip(used_cluster_alg, aquired_micro_f1_scores, aquired_macro_f1_scores, aquired_nmi_scores):
                # append settings of the run to lists
                preprocessing_types.append(used_preprocess)
                cnns.append(used_cnn)
                dr_techniques.append(used_dr)
                cluster_techniques.append(cl_alg)
                micro_f1_scores.append(micro_f1_score)
                macro_f1_scores.append(macro_f1_score)
                nmi_scores.append(nmi_score)
    # save results in pd
    dict = {'Preprocess': preprocessing_types,
               'CNN': cnns,
               'DR': dr_techniques,
               'Clustering': cluster_techniques,
               'Micro F1-Score': micro_f1_scores,
               'Macro F1-Score': macro_f1_scores,
               'NMI': nmi_scores}
    df = pd.DataFrame(dict)
    result_dir = Path(site_folder + 'results')
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    result_filename = 'Schiefer_analysis.csv'
    df.to_csv(result_dir / result_filename, index=False)

if __name__ == "__main__":
    main()