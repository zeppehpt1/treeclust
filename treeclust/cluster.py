import numpy as np
import pandas as pd
import umap as up
import pickle

from typing import Tuple
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.cluster import (
    KMeans,
    OPTICS,
    AgglomerativeClustering,
    AffinityPropagation,
    MeanShift,
    estimate_bandwidth,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    f1_score,
    v_measure_score,
    silhouette_score,
    matthews_corrcoef,
    cohen_kappa_score,
)
from sklearn.datasets import make_blobs
from joblib import Parallel, delayed
from fcmeans import FCM
from yellowbrick.cluster import KElbowVisualizer
from collections import Counter

import label_tools as lt
from constants import NUMBER_OF_CLASSES


def load_features(features_path: str) -> Tuple[list, NDArray, list]:
    with open(features_path, "rb") as f:
        data = pickle.load(f)
    files = data["filename"]
    features = data["features"]
    labels = data["labels"]
    return files, features, labels


def load_le(le_path: str):
    with open(le_path, "rb") as l:
        le = pickle.load(l)
    return le


def get_int_labels(features_path: str, le) -> list:
    _, _, labels = load_features(features_path)
    return le.transform(labels)


def load_all_data(features_path: str, le_path: str) -> Tuple[list, NDArray, list, list]:
    files, features, labels = load_features(features_path)
    le = load_le(le_path)
    y_gt = get_int_labels(features_path, le)
    return files, features, labels, y_gt


def pca(features: NDArray) -> NDArray:
    pca = PCA(n_components=0.95, svd_solver="full", whiten=True)
    pca.fit(features)
    reduced = pca.fit_transform(features)
    return reduced


def umap(features: NDArray, RANDOM_SEED: int) -> NDArray:
    reducer = up.UMAP(n_components=2, metric="cosine", random_state=RANDOM_SEED)
    reduced = reducer.fit_transform(features)
    return reduced


def tsne(features: NDArray, RANDOM_SEED: int) -> Tuple[NDArray, NDArray]:
    # no whitening
    pca_nw = PCA(
        n_components=50, svd_solver="full", whiten=False, random_state=RANDOM_SEED
    )
    x_nw = pca_nw.fit_transform(features)
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED)
    x_nw_tsne = tsne.fit_transform(x_nw)
    # with withening
    pca = PCA(n_components=50, svd_solver="full", whiten=True, random_state=RANDOM_SEED)
    x = pca.fit_transform(features)
    tsne_w = TSNE(n_components=2, random_state=RANDOM_SEED)
    x_w_tsne = tsne_w.fit_transform(x)
    return x_nw_tsne, x_w_tsne


def elbow_score(reduced_f: NDArray) -> float:  # distortion
    kmean_model = KMeans(n_init="auto", init="k-means++")
    visualizer = KElbowVisualizer(kmean_model, k=(2, 15), timings=True)
    visualizer.fit(reduced_f)  # Fit data to visualizer
    return visualizer.elbow_value_


def kMeansRes(scaled_data: NDArray, k: int, alpha_k=0.02) -> float:
    """
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
    """

    inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
    # fit k-means
    kmeans = KMeans(n_init="auto", init="k-means++", n_clusters=k).fit(scaled_data)
    scaled_inertia = kmeans.inertia_ / inertia_o + alpha_k * k
    return scaled_inertia


def chooseBestKforKMeansParallel(
    scaled_data: NDArray, k_range: int
) -> Tuple[int, DataFrame]:
    """
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
    """
    ans = Parallel(n_jobs=-1, verbose=10)(
        delayed(kMeansRes)(scaled_data, k) for k in k_range
    )
    ans = list(zip(k_range, ans))
    results = pd.DataFrame(ans, columns=["k", "Scaled Inertia"]).set_index("k")
    best_k = results.idxmin()[0]
    return best_k, results


def scaled_insertia(
    scaled_data: NDArray, k_range=range(2, 20)
) -> Tuple[int, DataFrame]:
    ans = []
    for k in k_range:
        scaled_inertia = kMeansRes(scaled_data, k)
        ans.append((k, scaled_inertia))
    results = pd.DataFrame(ans, columns=["k", "Scaled Inertia"]).set_index("k")
    best_k = results.idxmin()[0]
    return best_k, results


def ch_index(reduced_f: NDArray) -> int:
    model = KMeans(n_init="auto", init="k-means++")
    visualizer = KElbowVisualizer(
        model, k=(2, 15), metric="calinski_harabasz", timings=True
    )
    visualizer.fit(reduced_f)
    return visualizer.elbow_value_


def silhouette_score(reduced_f: NDArray) -> int:
    model = KMeans(n_init="auto", init="k-means++")
    visualizer = KElbowVisualizer(model, k=(2, 15), metric="silhouette", timings=True)
    visualizer.fit(reduced_f)
    return visualizer.elbow_value_


def affinity_propagation(reduced_f: NDArray) -> int:
    model = AffinityPropagation(preference=-50)
    model.fit(reduced_f)
    cluster_centers_indices = model.cluster_centers_indices_
    n_clusters = len(cluster_centers_indices)
    return n_clusters


def most_common_elem(lst: list) -> int:
    data = Counter(lst)
    return max(lst, key=data.get)


def get_proposed_cluster_numbers(reduced_f: NDArray) -> list:
    elbow_k = elbow_score(reduced_f)
    scaled_k, results = scaled_insertia(reduced_f)
    ch_index_k = ch_index(reduced_f)
    silhouette_k = silhouette_score(reduced_f)
    affinity_k = affinity_propagation(reduced_f)
    return [elbow_k, scaled_k, ch_index_k, silhouette_k, affinity_k]


def determine_best_k(reduced_f: NDArray) -> int:
    return most_common_elem(get_proposed_cluster_numbers(reduced_f))


def mean_shift(
    reduced_f: NDArray, y_gt: list, RANDOM_SEED: int
) -> Tuple[list, float, float, float, float, float, float, float, int]:
    bandwidth = estimate_bandwidth(
        reduced_f, quantile=0.3, n_samples=None, random_state=RANDOM_SEED
    )
    model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    model.fit(reduced_f)
    labels_unmatched = model.labels_
    y_pred = lt.label_matcher(labels_unmatched, y_gt)
    # cluster_centers = model.cluster_centers_
    # labels_unique = np.unique(labels)
    # n_clusters_ = len(labels_unique)
    (
        micro_f1_score,
        macro_f1_score,
        weighted_f1_score,
        nmi,
        f_star,
        cohen_kappa,
        mcc,
    ) = get_accuracy_value(y_gt, y_pred)
    num_pred_species = len(set(y_pred))
    return (
        y_pred,
        micro_f1_score,
        macro_f1_score,
        weighted_f1_score,
        f_star,
        cohen_kappa,
        mcc,
        nmi,
        num_pred_species,
    )


def k_means(
    reduced_f: NDArray, y_gt: list, RANDOM_SEED: int
) -> Tuple[list, float, float, float, float, float, float, float, int]:
    model = KMeans(
        n_clusters=NUMBER_OF_CLASSES,
        init="k-means++",
        n_init=500,
        random_state=RANDOM_SEED,
    )
    model.fit(reduced_f)
    labels_unmatched = model.labels_
    y_pred = lt.label_matcher(labels_unmatched, y_gt)
    (
        micro_f1_score,
        macro_f1_score,
        weighted_f1_score,
        nmi,
        f_star,
        cohen_kappa,
        mcc,
    ) = get_accuracy_value(y_gt, y_pred)
    num_pred_species = len(set(y_pred))
    return (
        y_pred,
        micro_f1_score,
        macro_f1_score,
        weighted_f1_score,
        f_star,
        cohen_kappa,
        mcc,
        nmi,
        num_pred_species,
    )


def agglo_cl(
    reduced_f: NDArray, y_gt: list, RANDOM_SEED: int
) -> Tuple[list, float, float, float, float, float, float, float, int]:
    model = AgglomerativeClustering(n_clusters=NUMBER_OF_CLASSES)
    model.fit(reduced_f)
    labels_unmatched = model.labels_
    y_pred = lt.label_matcher(labels_unmatched, y_gt)
    (
        micro_f1_score,
        macro_f1_score,
        weighted_f1_score,
        nmi,
        f_star,
        cohen_kappa,
        mcc,
    ) = get_accuracy_value(y_gt, y_pred)
    num_pred_species = len(set(y_pred))
    return (
        y_pred,
        micro_f1_score,
        macro_f1_score,
        weighted_f1_score,
        f_star,
        cohen_kappa,
        mcc,
        nmi,
        num_pred_species,
    )


def fuzzy_c_means(
    reduced_f: NDArray, y_gt: list, RANDOM_SEED: int
) -> Tuple[list, float, float, float, float, float, float, float, int]:
    model = FCM(n_clusters=NUMBER_OF_CLASSES, random_state=RANDOM_SEED)
    model.fit(reduced_f)
    labels_unmatched = model.predict(reduced_f)
    y_pred = lt.label_matcher(labels_unmatched, y_gt)
    (
        micro_f1_score,
        macro_f1_score,
        weighted_f1_score,
        nmi,
        f_star,
        cohen_kappa,
        mcc,
    ) = get_accuracy_value(y_gt, y_pred)
    num_pred_species = len(set(y_pred))
    return (
        y_pred,
        micro_f1_score,
        macro_f1_score,
        weighted_f1_score,
        f_star,
        cohen_kappa,
        mcc,
        nmi,
        num_pred_species,
    )


def optics(
    reduced_f: NDArray, y_gt, RANDOM_SEED: int
) -> Tuple[list, float, float, float, float, float, float, float, int]:
    model = OPTICS(min_samples=5).fit(reduced_f)
    labels_unmatched = model.labels_
    y_pred = lt.label_matcher(labels_unmatched, y_gt)
    (
        micro_f1_score,
        macro_f1_score,
        weighted_f1_score,
        nmi,
        f_star,
        cohen_kappa,
        mcc,
    ) = get_accuracy_value(y_gt, y_pred)
    num_pred_species = len(set(y_pred))
    return (
        y_pred,
        micro_f1_score,
        macro_f1_score,
        weighted_f1_score,
        f_star,
        cohen_kappa,
        mcc,
        nmi,
        num_pred_species,
    )


def get_cluster_res(
    reduced_f: NDArray, y_gt: list, RANDOM_SEED: int
) -> Tuple[list, list, list, list, list, list, list, list, list, list]:
    # TODO: make cluster algorithms flexible usable
    fns = [mean_shift, k_means, agglo_cl, fuzzy_c_means, optics]
    fns_idents = ["mean-shift", "k-means++", "agglo", "fc-means", "optics"]
    cluster_techniques = []
    micro_f1_scores = []
    macro_f1_scores = []
    weighted_f1_scores = []
    f_star_values = []
    cohen_kappa_scores = []
    nmi_scores = []
    mcc_scores = []
    y_pred_labels = []
    species_pred = []
    for fn, ident in zip(fns, fns_idents):
        (
            y_pred,
            micro_f1_score,
            macro_f1_score,
            weighted_f1_score,
            f_star,
            cohen_kappa,
            mcc,
            nmi,
            species,
        ) = fn(reduced_f, y_gt, RANDOM_SEED)
        cluster_techniques.append(ident)
        micro_f1_scores.append(micro_f1_score)
        macro_f1_scores.append(macro_f1_score)
        weighted_f1_scores.append(weighted_f1_score)
        f_star_values.append(f_star)
        cohen_kappa_scores.append(cohen_kappa)
        mcc_scores.append(mcc)
        nmi_scores.append(nmi)
        y_pred_labels.append(y_pred)
        species_pred.append(species)
    return (
        cluster_techniques,
        micro_f1_scores,
        macro_f1_scores,
        weighted_f1_scores,
        f_star_values,
        cohen_kappa_scores,
        mcc_scores,
        nmi_scores,
        y_pred_labels,
        species_pred,
    )


def tp(y_true, y_pred):
    return np.sum(np.multiply([i == True for i in y_pred], y_true))


def fp(y_true, y_pred):
    return np.sum(np.multiply([i == True for i in y_pred], [not (j) for j in y_true]))


def tn(y_true, y_pred):
    return np.sum(np.multiply([i == False for i in y_pred], [not (j) for j in y_true]))


def fn(y_true, y_pred):
    return np.sum(np.multiply([i == False for i in y_pred], y_true))


def get_multiclass_cm_values(y_true: list, y_pred: list):
    tp_values = []
    fp_values = []
    tn_values = []
    fn_values = []
    for i in np.unique(y_true):
        modified_true = [i == j for j in y_true]
        modified_pred = [i == j for j in y_pred]
        TP = tp(modified_true, modified_pred)
        tp_values.append(TP)
        FP = fp(modified_true, modified_pred)
        fp_values.append(FP)
        TN = tn(modified_true, modified_pred)
        tn_values.append(TN)
        FN = fn(modified_true, modified_pred)
        fn_values.append(FN)
    return (
        np.mean(tp_values),
        np.mean(fp_values),
        np.mean(fp_values),
        np.mean(fp_values),
    )


def f_star(y_true: list, y_pred: list):
    TP, FP, TN, FN = get_multiclass_cm_values(y_true, y_pred)
    return TP / (FN + FP + TP)


def get_accuracy_value(
    y_gt: list, y_pred: list
) -> Tuple[float, float, float, float, float, float, float]:
    return (
        f1_score(y_gt, y_pred, average="micro"),
        f1_score(y_gt, y_pred, average="macro"),
        f1_score(y_gt, y_pred, average="weighted"),
        v_measure_score(y_gt, y_pred),
        f_star(y_gt, y_pred),
        cohen_kappa_score(y_gt, y_pred),
        matthews_corrcoef(y_gt, y_pred),
    )


if __name__ == "__main__":
    # test data
    X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=12)
    print(get_proposed_cluster_numbers(X))
    print(determine_best_k(X))
