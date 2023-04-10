from matplotlib import gridspec, cm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import seaborn as sns
import pickle
import skimage.io
import yellowbrick
from skimage.feature import hog
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN, OPTICS, SpectralClustering, AgglomerativeClustering, FeatureAgglomeration, Birch, AffinityPropagation, MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA,KernelPCA, SparsePCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report, silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, rand_score, adjusted_rand_score, mutual_info_score, adjusted_mutual_info_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, fowlkes_mallows_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import make_blobs
from sklearn_extra.cluster import KMedoids
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from PIL import Image
from fcmeans import FCM
from yellowbrick.cluster import KElbowVisualizer
from gap_statistic import OptimalK
from collections import Counter
import umap as up

import statsmodels.api as sm
from scipy.stats import shapiro, kstest, normaltest, anderson

from prepare import RANDOM_SEED

import label_tools as lt

def load_features(features_path):
    with open(features_path, 'rb') as f:
        data = pickle.load(f)
    files = data['filename']
    features = data['features']
    labels = data['labels']
    return files, features, labels

def load_le(le_path):
    with open(le_path, 'rb') as l:
        le = pickle.load(l)
    return le

def get_int_labels(features_path, le):
    _,_,labels =  load_features(features_path)
    return le.transform(labels)

def load_all_data(features_path, le_path):
    files, features, labels = load_features(features_path)
    le = load_le(le_path)
    y_gt = get_int_labels(features_path, le)
    return files, features, labels, y_gt
    
def pca(features):
    pca = PCA(n_components=0.85, svd_solver='full', whiten=True)
    pca.fit(features)
    reduced = pca.fit_transform(features)
    return reduced

# !!! no randomseed!
def umap(features):
    reducer = up.UMAP(n_components=2, metric='cosine')
    reduced = reducer.fit_transform(features)
    return reduced

def tsne(features):
    # no whitening
    pca_nw = PCA(n_components=50, svd_solver='full', whiten=False, random_state=RANDOM_SEED)
    x_nw = pca_nw.fit_transform(features)
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED)
    x_nw_tsne = tsne.fit_transform(x_nw)
    #with withening
    pca = PCA(n_components=50, svd_solver='full', whiten=True, random_state=RANDOM_SEED)
    x = pca.fit_transform(features)
    tsne_w = TSNE(n_components=2, random_state=RANDOM_SEED)
    x_w_tsne = tsne_w.fit_transform(x)
    return x_nw_tsne, x_w_tsne

# determine number of k
# TODO: create multiple settings, store results and use the value that has the majority suggestions
def elbow_score(reduced_f): # distortion
    kmean_model = KMeans(init='k-means++', random_state=RANDOM_SEED)
    visualizer = KElbowVisualizer(kmean_model, k=(2, 15), timings= True)
    visualizer.fit(reduced_f)        # Fit data to visualizer
    return visualizer.elbow_value_

def kMeansRes(scaled_data, k, alpha_k=0.02):
    '''
    Parameters 
    ----------
    scaled_data: matrix 
        scaled data. rows are samples and columns are features for clustering
    k: int
        current k for applying KMeans
    alpha_k: float
        manually tuned factor that gives penalty to the number of clusters
    Returns 
    -------
    scaled_inertia: float
        scaled inertia value for current k           
    '''
    
    inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
    # fit k-means
    kmeans = KMeans(n_init='auto',init='k-means++', n_clusters=k, random_state=RANDOM_SEED).fit(scaled_data)
    scaled_inertia = kmeans.inertia_ / inertia_o + alpha_k * k
    return scaled_inertia
def chooseBestKforKMeansParallel(scaled_data, k_range):
    '''
    Parameters 
    ----------
    scaled_data: matrix 
        scaled data. rows are samples and columns are features for clustering
    k_range: list of integers
        k range for applying KMeans
    Returns 
    -------
    best_k: int
        chosen value of k out of the given k range.
        chosen k is k with the minimum scaled inertia value.
    results: pandas DataFrame
        adjusted inertia value for each k in k_range
    '''
    ans = Parallel(n_jobs=-1,verbose=10)(delayed(kMeansRes)(scaled_data, k) for k in k_range)
    ans = list(zip(k_range,ans))
    results = pd.DataFrame(ans, columns = ['k','Scaled Inertia']).set_index('k')
    best_k = results.idxmin()[0]
    return best_k, results
def scaled_insertia(scaled_data, k_range=range(2, 20)):
    ans = []
    for k in k_range:
        scaled_inertia = kMeansRes(scaled_data, k)
        ans.append((k, scaled_inertia))
    results = pd.DataFrame(ans, columns = ['k','Scaled Inertia']).set_index('k')
    best_k = results.idxmin()[0]
    return best_k, results

def gap_statistic(reduced_f):
    optimalK = OptimalK(n_jobs=4, parallel_backend='joblib')
    n_clusters = optimalK(reduced_f, cluster_array=np.arange(2, 15))
    return n_clusters

def ch_index(reduced_f):
    model = KMeans(init='k-means++', random_state=RANDOM_SEED)
    visualizer = KElbowVisualizer(model, k=(2, 15), metric='calinski_harabasz', timings=True)
    visualizer.fit(reduced_f)
    return visualizer.elbow_value_

def silhouette_score(reduced_f):
    model = KMeans(init='k-means++', random_state=RANDOM_SEED)
    visualizer = KElbowVisualizer(model, k=(2, 15), metric='silhouette', timings=True)
    visualizer.fit(reduced_f)
    return visualizer.elbow_value_

def affinity_propagation(reduced_f):
    model = AffinityPropagation(preference=-50)
    model.fit(reduced_f)
    cluster_centers_indices = model.cluster_centers_indices_
    n_clusters = len(cluster_centers_indices)
    return n_clusters

def most_common_elem(lst):
    data = Counter(lst)
    return max(lst, key=data.get)

def determine_best_k(reduced_f):
    elbow_k = elbow_score(reduced_f)
    scaled_k, results = scaled_insertia(reduced_f)
    gap_k = gap_statistic(reduced_f)
    ch_index_k = ch_index(reduced_f)
    silhouette_k = silhouette_score(reduced_f)
    affinity_k = affinity_propagation(reduced_f)
    cluster_proposals = [elbow_k, scaled_k, gap_k, ch_index_k, silhouette_k, affinity_k]
    optimal_k = most_common_elem(cluster_proposals)
    return optimal_k

def mean_shift(reduced_f, labels, y_gt):
    bandwidth = estimate_bandwidth(reduced_f, quantile=0.1, n_samples=500, random_state=RANDOM_SEED)
    model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    model.fit(reduced_f)
    labels_unmatched = model.labels_
    y_pred = lt.label_matcher(labels_unmatched, y_gt)
    cluster_centers = model.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    micro_f1_score, macro_f1_score, nmi = get_accuracy_value(y_gt, y_pred)
    return y_pred, micro_f1_score, macro_f1_score, nmi

def k_means(reduced_f, y_gt):
    model = KMeans(n_clusters=4, init='k-means++', n_init=500, random_state=RANDOM_SEED)
    model.fit(reduced_f)
    labels_unmatched = model.labels_
    y_pred = lt.label_matcher(labels_unmatched, y_gt)
    micro_f1_score, macro_f1_score, nmi = get_accuracy_value(y_gt, y_pred)
    return y_pred, micro_f1_score, macro_f1_score, nmi

def agglo_cl(reduced_f, y_gt):
    model = AgglomerativeClustering(n_clusters=4)
    model.fit(reduced_f)
    labels_unmatched = model.labels_
    y_pred = lt.label_matcher(labels_unmatched, y_gt)
    micro_f1_score, macro_f1_score, nmi = get_accuracy_value(y_gt, y_pred)
    return y_pred, micro_f1_score, macro_f1_score, nmi

def run_cluster(reduced_f, labels, y_gt):
    cluster_techniques = []
    micro_f1_scores = []
    macro_f1_scores = []
    nmi_scores = []
    ms_y_pred, ms_micro_f1_score, ms_macro_f1_score, ms_nmi = mean_shift(reduced_f, labels, y_gt)
    ms_ident_str = 'mean-shift'
    cluster_techniques.append(ms_ident_str)
    micro_f1_scores.append(ms_micro_f1_score)
    macro_f1_scores.append(ms_macro_f1_score)
    nmi_scores.append(ms_nmi)
    km_y_pred, km_micro_f1_score, km_macro_f1_score, km_nmi = k_means(reduced_f, y_gt)
    km_ident_str = 'k-means++'
    cluster_techniques.append(km_ident_str)
    micro_f1_scores.append(km_micro_f1_score)
    macro_f1_scores.append(km_macro_f1_score)
    nmi_scores.append(km_nmi)
    agglo_y_pred, agglo_micro_f1_score, agglo_macro_f1_score, agglo_nmi = k_means(reduced_f, y_gt)
    agglo_ident_str = 'agglo'
    cluster_techniques.append(agglo_ident_str)
    micro_f1_scores.append(agglo_micro_f1_score)
    macro_f1_scores.append(agglo_macro_f1_score)
    nmi_scores.append(agglo_nmi)
    return cluster_techniques, micro_f1_scores, macro_f1_scores, nmi_scores

def get_accuracy_value(y_gt, y_pred):
    micro_f1_score = f1_score(y_gt, y_pred, average='micro')
    macro_f1_score = f1_score(y_gt, y_pred, average='macro')
    nmi = v_measure_score(y_gt, y_pred)
    return micro_f1_score, macro_f1_score, nmi

if __name__ == "__main__":
    # test data
    X, y = make_blobs(n_samples=80, centers=3, n_features=2, random_state=RANDOM_SEED)
    print(determine_best_k(X))