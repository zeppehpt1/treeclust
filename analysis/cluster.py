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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn_extra.cluster import KMedoids
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from PIL import Image
from fcmeans import FCM

import statsmodels.api as sm
from scipy.stats import shapiro, kstest, normaltest, anderson

from analysis import label_tools as lt

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
    gt_le = get_int_labels(features_path, le)
    return files, features, labels, gt_le
    
def pca(features):
    pca = PCA(n_components=0.85, svd_solver='full', random_state=42, whiten=True)
    pca.fit(features)
    reduced = pca.fit_transform(features)
    return reduced

def umap(features):
    reducer = umap.UMAP(n_components=2, metric='cosine', random_state=666)
    reduced = reducer.fit_transform(features)
    return reduced

def tsne(features):
    # no whitening
    pca_nw = PCA(n_components=50, svd_solver='full', whiten=False, random_state=42)
    x_nw = pca_nw.fit_transform(features)
    tsne = TSNE(n_components=2, random_state=567)
    x_nw_tsne = tsne.fit_transform(x_nw)
    #with withening
    pca = PCA(n_components=50, svd_solver='full', whiten=True, random_state=42)
    x = pca.fit_transform(features)
    tsne_w = TSNE(n_components=2, random_state=567)
    x_w_tsne = tsne_w.fit_transform(x)
    return x_nw_tsne, x_w_tsne

def mean_shift(reduced_f):
    bandwidth = estimate_bandwidth(reduced_f, quantile=0.3, n_samples=7, random_state=444)
    model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    model.fit(reduced_f)
    
    labels_unmatched = model.labels_
    y_pred = lt.label_matcher(labels_unmatched, y_gt)
    cluster_centers = model.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)