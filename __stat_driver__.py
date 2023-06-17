# use functions to collect all 50 means for statistical evaluation from every dataset in a single module

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro
import glob
import pickle
import umap
import random
import seaborn as sns
import importlib
from random import randint
from typing import final
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import f1_score
from fcmeans import FCM

from analysis import label_tools as lt
from analysis.label_tools import CustomLabelEncoder
from analysis import evaluation
from analysis.evaluation import calc_multiple_means
from analysis.constants import unique_rand

importlib.reload(evaluation)

random.seed(698)
RANDOM_SEEDS = unique_rand(1, 999999, 1000)
#print(RANDOM_SEEDS)

# 1
# fc1_path = Path('data/Schiefer/encodings/densenet_clahe_polygon_pred.pickle')
# le_path = Path('data/Schiefer/label_encodings/densenet_clahe_label_encodings.pickle')
# best = calc_multiple_means(fc1_path, le_path, RANDOM_SEEDS, 20, 'pca', 10, 'k-means++')

# 2
# fc1_path = Path('data/Schiefer/encodings/densenet_clahe-denoising_polygon_pred.pickle')
# le_path = Path('data/Schiefer/label_encodings/densenet_clahe-denoising_label_encodings.pickle')
# second = calc_multiple_means(fc1_path, le_path, RANDOM_SEEDS, 20, 'pca', 10, 'k-means++')

# 3
# fc1_path = Path('data/Schiefer/encodings/resnet_clahe_polygon_pred.pickle')
# le_path = Path('data/Schiefer/label_encodings/resnet_clahe_label_encodings.pickle')
# third = calc_multiple_means(fc1_path, le_path, RANDOM_SEEDS, 20, 'umap', 10, 'fc-means')


# 4
# fc1_path = Path('data/Schiefer/encodings/resnet_clahe_polygon_pred.pickle')
# le_path = Path('data/Schiefer/label_encodings/resnet_clahe_label_encodings.pickle')
# fourth = calc_multiple_means(fc1_path, le_path, RANDOM_SEEDS, 20, 'umap', 10, 'k-means++')

# 5
# fc1_path = Path('data/Schiefer/encodings/resnet_clahe_polygon_pred.pickle')
# le_path = Path('data/Schiefer/label_encodings/resnet_clahe_label_encodings.pickle')
# fifth = calc_multiple_means(fc1_path, le_path, RANDOM_SEEDS, 20, 'umap', 10, 'agglo')

# schiefer_runs = [best, second, third, fourth, fifth]
# with open('data/schiefer_res_50_means_V2', 'wb') as f:
#     pickle.dump(schiefer_runs, f)

############################################

# 1
best = calc_multiple_means('/home/richard/data/Bamberg_Stadtwald/encodings/resnet_clahe_polygon_pred.pickle', '/home/richard/data/Bamberg_Stadtwald/label_encodings/resnet_clahe_label_encodings.pickle', RANDOM_SEEDS, 20, 'umap', 9, 'optics')

# 2
second = calc_multiple_means('/home/richard/data/Bamberg_Stadtwald/encodings/resnet_clahe-denoising_polygon_pred.pickle', '/home/richard/data/Bamberg_Stadtwald/label_encodings/resnet_clahe-denoising_label_encodings.pickle', RANDOM_SEEDS, 20, 'umap', 9, 'optics')

# 3
third = calc_multiple_means('/home/richard/data/Bamberg_Stadtwald/encodings/resnet_clahe_polygon_pred.pickle', '/home/richard/data/Bamberg_Stadtwald/label_encodings/resnet_clahe_label_encodings.pickle', RANDOM_SEEDS, 20, 'umap', 9, 'k-means++')

# 4
fourth = calc_multiple_means('/home/richard/data/Bamberg_Stadtwald/encodings/resnet_clahe_polygon_pred.pickle', '/home/richard/data/Bamberg_Stadtwald/label_encodings/resnet_clahe_label_encodings.pickle', RANDOM_SEEDS, 20, 'umap', 9, 'agglo')

# 5
fifth = calc_multiple_means('/home/richard/data/Bamberg_Stadtwald/encodings/vgg_clahe_polygon_pred.pickle', '/home/richard/data/Bamberg_Stadtwald/label_encodings/vgg_clahe_label_encodings.pickle', RANDOM_SEEDS, 20, 'umap', 9, 'optics')

stadtwald_runs = [best, second, third, fourth, fifth]
with open('data/stadtwald_res_50_means_V2', 'wb') as f:
    pickle.dump(stadtwald_runs, f)
print("Stadtwald analyzed")
    
################################################

# 1
best = calc_multiple_means('/home/richard/data/Tretzendorf/encodings/resnet_clahe-denoising_polygon_pred.pickle', '/home/richard/data/Tretzendorf/label_encodings/resnet_clahe-denoising_label_encodings.pickle', RANDOM_SEEDS, 20, 'umap', 8, 'k-means++')

# 2
second = calc_multiple_means('/home/richard/data/Tretzendorf/encodings/resnet_clahe-denoising_polygon_pred.pickle', '/home/richard/data/Tretzendorf/label_encodings/resnet_clahe-denoising_label_encodings.pickle', RANDOM_SEEDS, 20, 'umap', 8, 'fc-means')

# 3
third = calc_multiple_means('/home/richard/data/Tretzendorf/encodings/densenet_clahe-denoising_polygon_pred.pickle', '/home/richard/data/Tretzendorf/label_encodings/densenet_clahe-denoising_label_encodings.pickle', RANDOM_SEEDS, 20, 'umap', 8, 'fc-means')

# 4
fourth = calc_multiple_means('/home/richard/data/Tretzendorf/encodings/densenet_clahe-denoising_polygon_pred.pickle', '/home/richard/data/Tretzendorf/label_encodings/densenet_clahe-denoising_label_encodings.pickle', RANDOM_SEEDS, 20, 'umap', 8, 'k-means++')

# 5
fifth = calc_multiple_means('/home/richard/data/Tretzendorf/encodings/resnet_clahe-denoising_polygon_pred.pickle', '/home/richard/data/Tretzendorf/label_encodings/resnet_clahe-denoising_label_encodings.pickle', RANDOM_SEEDS, 20, 'umap', 8, 'agglo')

tretzendorf_runs = [best, second, third, fourth, fifth]
with open('data/tretzendorf_res_50_means_V2', 'wb') as f:
    pickle.dump(tretzendorf_runs, f)
print("Tretzendorf analyzed")