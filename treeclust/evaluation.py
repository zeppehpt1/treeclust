import pickle
import umap
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Tuple, Union
from geopandas import GeoDataFrame
from pathlib import Path, PosixPath
from tqdm import tqdm
from fcmeans import FCM
from scipy.stats import shapiro
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

import label_tools as lt
from constants import SITE

def get_files(csv_dir: Union[str, PosixPath]) -> list:
    return sorted(csv_dir.glob(('*_' + SITE + '_analysis.pickle')))

def get_basic_df(csv_dir: Union[str, PosixPath]) -> DataFrame:
    csv_files = get_files(csv_dir)
    df = pd.read_pickle(csv_files[0]) # beware change!
    new_df = df.get(['Preprocess', 'CNN', 'DR', 'Clustering']) # or include pred labels with 'Pred labels'
    return new_df

def get_single_column_dfs(csv_dir: Union[str, PosixPath], column_name:str) -> list:
    csv_files = get_files(csv_dir)
    dfs = []
    for csv in csv_files:
        #df = pd.read_csv(csv)
        df = pd.read_pickle(csv)
        dfs.append(df[column_name])
    return dfs

def get_mean_values(dfs:list) -> Union[float, int]:
    divisor = len(dfs)
    sum_column = sum(dfs) / divisor
    return sum_column

def get_pred_spec_mean(csv_dir: Union[str, PosixPath]) -> Tuple[list, str]:
    column_name = 'Pred species'
    dfs = get_single_column_dfs(csv_dir, column_name)
    column = round(get_mean_values(dfs))
    new_c_name = 'Mean(r) species'
    return column, new_c_name

def get_micro_f1_mean(csv_dir: Union[str, PosixPath]) -> Tuple[list, str]: # is essentially 'accuracy' in a multi-class scenario
    column_name = 'Micro F1-Score'
    dfs = get_single_column_dfs(csv_dir, column_name)
    column = get_mean_values(dfs)
    new_c_name = 'Mean Micro F1-Score'
    return column, new_c_name

def get_macro_f1_mean(csv_dir: Union[str, PosixPath]) -> Tuple[list, str]:
    column_name = 'Macro F1-Score'
    dfs = get_single_column_dfs(csv_dir, column_name)
    column = get_mean_values(dfs)
    new_c_name = 'Mean Macro F1-Score'
    return column, new_c_name

def get_weighted_f1_mean(csv_dir: Union[str, PosixPath]) -> Tuple[list, str]:
    column_name = 'Weighted F1-Score'
    dfs = get_single_column_dfs(csv_dir, column_name)
    column = get_mean_values(dfs)
    new_c_name = 'Mean Weighted F1-Score'
    return column, new_c_name

def get_f_star_mean(csv_dir: Union[str, PosixPath]) -> Tuple[list, str]:
    column_name = 'F-Star Score'
    dfs = get_single_column_dfs(csv_dir, column_name)
    column = get_mean_values(dfs)
    new_c_name = 'Mean F-Star Score'
    return column, new_c_name

def get_cohen_kappa_mean(csv_dir: Union[str, PosixPath]) -> Tuple[list, str]:
    column_name = 'Cohens Kappa'
    dfs = get_single_column_dfs(csv_dir, column_name)
    column = get_mean_values(dfs)
    new_c_name = 'Mean Cohens Kappa'
    return column, new_c_name

def get_mcc_mean(csv_dir: Union[str, PosixPath]) -> Tuple[list, str]:
    column_name = 'Matthews correlation coefficient'
    dfs = get_single_column_dfs(csv_dir, column_name)
    column = get_mean_values(dfs)
    new_c_name = 'Mean Matthews correlation coefficient'
    return column, new_c_name

def get_nmi_mean(csv_dir: Union[str, PosixPath]) -> Tuple[list, str]:
    column_name = 'NMI'
    dfs = get_single_column_dfs(csv_dir, column_name)
    column = get_mean_values(dfs)
    new_c_name = 'Mean NMI'
    return column, new_c_name

def get_complete_mean_df(csv_dir: Union[str, PosixPath]) -> DataFrame:
    new_df = get_basic_df(csv_dir)
    pred_mean, pred_name = get_pred_spec_mean(csv_dir)
    new_df[pred_name] = pred_mean
    micro_mean, micro_name = get_micro_f1_mean(csv_dir)
    new_df[micro_name] = micro_mean
    macro_mean, macro_name = get_macro_f1_mean(csv_dir)
    new_df[macro_name] = macro_mean
    weighted_mean, weighted_name = get_weighted_f1_mean(csv_dir)
    new_df[weighted_name] = weighted_mean
    f_star_mean, f_star_name = get_f_star_mean(csv_dir)
    new_df[f_star_name] = f_star_mean
    cohen_kappa_mean, cohen_kappa_name = get_cohen_kappa_mean(csv_dir)
    new_df[cohen_kappa_name] = cohen_kappa_mean
    mcc_mean, mcc_name = get_mcc_mean(csv_dir)
    new_df[mcc_name] = mcc_mean
    nmi_mean, nmi_name = get_nmi_mean(csv_dir)
    new_df[nmi_name] = nmi_mean
    return new_df

def calc_multiple_means(encoding_path: Union[str, PosixPath], le_path: Union[str, PosixPath], RANDOM_SEEDS:list, n_interval:int, dr:str, num_clusters:int, cluster_alg:str) -> list:
    assert Path(encoding_path).is_file(), "encoding file should exist"
    assert Path(le_path).is_file(), "label encoding file should exist"
    with open(encoding_path, 'rb') as f:
        data = pickle.load(f)
    with open(le_path, 'rb') as l:
        le = pickle.load(l)
    fc1 = data['features']
    labels = data['labels']
    y_gt = le.transform(labels)
    interval_values = []
    all_means = []
    reduced = 0 # only for initilization purposes
    if dr == 'pca':
        pca = PCA(n_components=0.95, svd_solver='full', whiten=True)
        pca.fit(fc1)
        reduced = pca.fit_transform(fc1)
    for SEED in (pbar := tqdm(range(len(RANDOM_SEEDS)))):
        pbar.set_description(f"Processing number {SEED}")
        if dr == 'umap':
            reducer = umap.UMAP(n_components=2, metric='cosine', random_state=RANDOM_SEEDS[SEED])
            reduced = reducer.fit_transform(fc1)
        if cluster_alg == 'k-means++':
            model = KMeans(n_clusters=num_clusters, init='k-means++', n_init=500, random_state=RANDOM_SEEDS[SEED])
            model.fit(reduced)
            labels_unmatched = model.labels_
        elif cluster_alg == 'fc-means':
            model = FCM(n_clusters=num_clusters, random_state=RANDOM_SEEDS[SEED])
            model.fit(reduced)
            labels_unmatched = model.predict(reduced)
        elif cluster_alg == 'agglo':
            model = AgglomerativeClustering(n_clusters=num_clusters)
            model.fit(reduced)
            labels_unmatched = model.labels_
        elif cluster_alg == 'optics':
            model = OPTICS(min_samples=5).fit(reduced)
            labels_unmatched = model.labels_
        y_pred = lt.label_matcher(labels_unmatched, y_gt)
        interval_values.append(f1_score(y_gt, y_pred, average='micro'))
        SEED = SEED +1
        if SEED % n_interval == 0:
            mean_res = np.mean(interval_values)
            all_means.append(mean_res)
            interval_values = []
            SEED = SEED -1
    if shapiro(all_means)[1] > 0.05:
        print("data follows normal distribution")
    else:
        print("no normal distribution!")
    return all_means

def calculate_iou(polygon1, polygon2) -> float:
    return polygon1.intersection(polygon2).area / polygon1.union(polygon2).area

def calculate_mean_iou_for_multiple_polygons(ground_truth_df:GeoDataFrame, prediction_df:GeoDataFrame) -> Tuple[float, int]:
    iou_scores = []
    errors = 0
    for gt_poly in ground_truth_df['geometry']:
        for pred_poly in prediction_df['geometry']:
            try:
                iou_score = calculate_iou(gt_poly, pred_poly)
                if iou_score != 0.0 and iou_score > 0.4:
                    iou_scores.append(iou_score)
            except:
                errors = errors + 1
    mean_iou_score = sum(iou_scores) / len(iou_scores)
    return mean_iou_score, len(iou_scores)

# tp = number of correct crowns above the threshold
def tp(iou_scores):
    return iou_scores

# fp = number of crowns below the threshold (number of predicted crowns - number of iou scores above threshold)
def fp(predicted_crowns, iou_scores):
    return predicted_crowns - iou_scores

# fn = number of missing crowns (number of ground truth crowns - number of iou scores above threshold)
def fn(ground_truth_crowns, iou_scores):
    return ground_truth_crowns - iou_scores

def precision(tp, fp):
    return tp / (tp + fp)

def recall(tp, fn):
    return tp / (tp + fn)

def object_detection_f1_score(precision, recall):
    return 2 * precision * recall / (precision + recall)

def calculate_object_detection_f1_score(ground_truth_df:GeoDataFrame, prediction_df:GeoDataFrame) -> float:
    """Calculate the f1-score of a corresponding test and training set. Preferably using only matching test and prediction tiles.

    Args:
        ground_truth_df (GeoDataFrame): Contains annotated ground truth data
        prediction_df (GeoDataFrame): Contains predictions of the test set

    Returns:
        float: f1-score
    """
    count_ground_truth_crowns = len(ground_truth_df)
    count_predicted_crowns = len(prediction_df)
    
    _, count_iou_scores = calculate_mean_iou_for_multiple_polygons(ground_truth_df, prediction_df)
    
    TP = tp(count_iou_scores)
    FP = fp(count_predicted_crowns, count_iou_scores)
    FN = fn(count_ground_truth_crowns, count_iou_scores)
    
    prec = precision(TP, FP)
    rec = recall(TP, FN)
    
    return object_detection_f1_score(prec, rec)