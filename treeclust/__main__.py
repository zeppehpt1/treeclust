import prepare
import preprocessing
import features as ft
import cluster
import evaluation
from constants import RANDOM_SEEDS, SITE

import pandas as pd
import os

from pathlib import Path
from tqdm import tqdm

def main():
    """Complete pipeline driver, which includes the following steps:
    
    Need to define paths to the orthomosaic tiles dir, ground truth masks dir and the corresponding shapefile dir. Or skip steps 1-4 if the .pickle files are available in the data folder. The whole pipeline creates .pickle files in the result dir with the clustering information.
    
    Caution: Orthomosaic tiles and shapefiles should have matching numbers for the pipeline to work properly.
    
    1. Extraction of single tree crown images
    2. Preprocessing of extracted images
    3. Feature extraction (encoding)
    4. Dimensionality reduction
    5. Clustering
    """
    
    # #schiefer
    site_folder = '/home/richard/data/' + SITE + '/'
    orthophoto_dir = site_folder + 'ortho_tiles/'
    gt_mask_dir = site_folder + 'gt_masks/'
    prediction_dir = site_folder + 'predictions/pred_crown_tiles/'
    # gt_annotations_dir = "" # optional
    
    # stadtwald & tretzendorf
    # site_folder = '/home/richard/data/' + SITE + '/'
    # orthophoto_dir = site_folder + 'ortho_tiles_aoi/'
    # gt_mask_dir = site_folder + 'gt_masks/'
    # prediction_dir = site_folder + 'predictions/pred_crown_tiles_gt_aois/'
    # gt_annotations_dir = "" # optional
    
    # stadtwald test set
    # site_folder = '/home/richard/data/' + SITE + '/'
    # orthophoto_dir = site_folder + 'ortho_tiles_aoi/'
    # gt_mask_dir = site_folder + 'gt_masks/'
    # prediction_dir = site_folder + 'predictions/pred_crown_tiles_gt_aois/'
    
    # define folders
    encoding_dir = Path(site_folder + 'encodings')
    label_encoding_dir = Path(site_folder + 'label_encodings')
    result_dir = Path(site_folder + 'results')
    Path(encoding_dir).mkdir(parents=True, exist_ok=True)
    Path(label_encoding_dir).mkdir(parents=True, exist_ok=True)
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    
    assert Path(site_folder).exists()
    assert Path(orthophoto_dir).exists()
    assert Path(gt_mask_dir).exists()
    assert Path(prediction_dir).exists()
    assert Path(encoding_dir).exists()
    assert Path(label_encoding_dir).exists()
    assert Path(result_dir).exists()

    # 1. extract single tree crowns (polygons) from orthophoto
    if not Path(site_folder + 'pred_polygon_clipped_raster_files').exists():
        Path(site_folder + 'pred_polygon_clipped_raster_files').mkdir(parents=True, exist_ok=True)
        prepare.clip_crown_sets_with_gt_masks(orthophoto_dir, prediction_dir, gt_mask_dir, make_squares=False, step_size=0.5)
    else:
        print("Cropped files already exist")

    # 2. preprocess images create two sets
    first_set = preprocessing.preprocess_images(site_folder + 'pred_polygon_clipped_raster_files', None, None, False, True, True, False)
    second_set = preprocessing.preprocess_images(site_folder + 'pred_polygon_clipped_raster_files', None, None, False, True, False, False)

    # 3. feature extraction
    cnns = ['vgg', 'resnet', 'effnet', 'densenet', 'inception']
    encoding_files = sorted(encoding_dir.glob('*.pickle'))
    if len(encoding_files) != len(cnns) * 2:
        for cnn in cnns:
            clahe_denoising = ft.create_and_save_le_encodings(cnn, first_set, site_folder)
            clahe = ft.create_and_save_le_encodings(cnn, second_set, site_folder)
    else:
        print("Encodings already exist")
    
    # analysis
    # 5 runs
    final_res_name ='final' + '_' + SITE + '_' + 'analysis.pickle'
    if os.path.exists(result_dir / final_res_name) == False:
        for RANDOM_SEED in (pbar := tqdm(RANDOM_SEEDS)):
            pbar.set_description(f"Run analysis with seed {RANDOM_SEED}")
            
            # define lists for csv dump
            preprocessing_types = []
            cnns = []
            dr_techniques = []
            cluster_techniques = []
            micro_f1_scores = []
            macro_f1_scores = []
            weighted_f1_scores = []
            nmi_scores = []
            y_pred_label_sets = []
            species_pred = []
            f_star_scores = []
            cohen_kappa_scores = []
            mcc_scores = []
            #pred_cluster_numbers = []
            
            run_ident = str(RANDOM_SEEDS.index(RANDOM_SEED))
            encoding_files = sorted(encoding_dir.glob('*.pickle'))
            le_files = sorted(label_encoding_dir.glob('*.pickle'))
            for encoding_file, le_file in zip(encoding_files, le_files):
                files, features, labels, y_gt = cluster.load_all_data(encoding_file, le_file)
                # 4. DR
                pca_reduced = cluster.pca(features)
                umap_reduced = cluster.umap(features, RANDOM_SEED)
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
                    # cluser analysis
                    # best k
                    #best_k = cluster.determine_best_k(dr_features)
                    # 5. clustering
                    used_cluster_alg, aquired_micro_f1_scores, aquired_macro_f1_scores, aquired_weighted_f1_scores, aquired_f_star_scores, aquired_cohen_kappa_scores, aquired_mcc_scores, aquired_nmi_scores, aquired_y_pred_label_sets, aquired_num_pred_species = cluster.get_cluster_res(dr_features, y_gt, RANDOM_SEED)
                    for cl_alg, micro_f1_score, macro_f1_score, weighted_f1_score, f_star_score, cohen_kappa_score, mcc_score, nmi_score, y_pred_label_set, pred_species_num in zip(used_cluster_alg, aquired_micro_f1_scores, aquired_macro_f1_scores, aquired_weighted_f1_scores, aquired_f_star_scores, aquired_cohen_kappa_scores, aquired_mcc_scores, aquired_nmi_scores, aquired_y_pred_label_sets, aquired_num_pred_species):
                        # append settings of the run to lists
                        preprocessing_types.append(used_preprocess)
                        cnns.append(used_cnn)
                        dr_techniques.append(used_dr)
                        cluster_techniques.append(cl_alg)
                        micro_f1_scores.append(micro_f1_score)
                        macro_f1_scores.append(macro_f1_score)
                        weighted_f1_scores.append(weighted_f1_score)
                        f_star_scores.append(f_star_score)
                        cohen_kappa_scores.append(cohen_kappa_score)
                        mcc_scores.append(mcc_score)
                        nmi_scores.append(nmi_score)
                        y_pred_label_sets.append(y_pred_label_set)
                        species_pred.append(pred_species_num)
                        #pred_cluster_numbers.append(best_k)
            
            # save results as pd
            results = {'Preprocess': preprocessing_types,
                    'CNN': cnns,
                    'DR': dr_techniques,
                    'Clustering': cluster_techniques,
                    'Micro F1-Score': micro_f1_scores,
                    'Macro F1-Score': macro_f1_scores,
                    'Weighted F1-Score': weighted_f1_scores,
                    'F-Star Score': f_star_scores,
                    'Cohens Kappa': cohen_kappa_scores,
                    'Matthews correlation coefficient': mcc_scores,
                    'NMI': nmi_scores,
                    'Pred labels': y_pred_label_sets,
                    'Pred species': species_pred}
                    #'Expected cluster number': pred_cluster_numbers}
            df = pd.DataFrame(results)
            
            result_filename = run_ident + '_' + SITE + '_' + 'analysis.pickle'
            pd.to_pickle(df, result_dir / result_filename)
            results = results.clear()
        final_df = evaluation.get_complete_mean_df(result_dir)
        pd.to_pickle(final_df, result_dir / final_res_name)
    else:
        print("Final results already produced")

if __name__ == "__main__":
    main()