import skimage
import pickle
import umap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from . import cluster
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image


def pretty_cm(cm, labelnames, cscale=0.6, ax0=None, fs=5, cmap='cool'):
    """
    Generates a pretty-formated confusion matrix for convenient visualization.
    
    The true labels are displayed on the rows, and the predicted labels are displayed on the columns.
    
    Parameters
    ----------
    cm: ndarray 
        nxn array containing the data of the confusion matrix.
    
    labelnames: list(string)
        list of class names in order on which they appear in the confusion matrix. For example, the first
        element should contain the class corresponding to the first row and column of *cm*.
    cscale: float
        parameter that adjusts the color intensity. Allows color to be present for confusion matrices with few mistakes,
        and controlling the intensity for ones with many misclassifications.
    
    ax0: None or matplotlib axis object
        if None, a new figure and axis will be created and the visualization will be displayed.
        if an axis is supplied, the confusion matrix will be plotted on the axis in place.
    fs: int
        font size for text on confusion matrix.
        
    cmap: str
        matplotlib colormap to use
    
    Returns
    ---------
    None
    
    """
    
    acc = cm.trace() / cm.sum()
    if ax0 is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 2.5), dpi=300)
        fig.set_facecolor('w')
    else:
        ax = ax0

    n = len(labelnames)
    ax.imshow(np.power(cm, cscale), cmap=cmap, extent=(0, n, 0, n))
    labelticks = np.arange(n) + 0.5
    
    ax.set_xticks(labelticks, minor=True)
    ax.set_yticks(labelticks, minor=True)
    ax.set_xticklabels(['' for i in range(n)], minor=False, fontsize=fs)
    ax.set_yticklabels(['' for i in range(n)], minor=False, fontsize=fs)
    
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels=labelnames, minor=True, fontsize=fs, rotation=90)
    ax.set_yticklabels(labels=reversed(labelnames), minor=True, fontsize=fs)

    ax.set_xlabel('Predicted Labels', fontsize=fs)
    ax.xaxis.set_label_position('bottom')
    
    ax.set_ylabel('Actual Labels', fontsize=fs)
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j + 0.5, n - i - 0.5, '{:^5}'.format(z), ha='center', va='center', fontsize=fs,
                bbox=dict(boxstyle='round', facecolor='w', edgecolor='0.3'))
    ax.grid(which='major', color=np.ones(3) * 0.33, linewidth=1)

    if ax0 is None:
        ax.set_title('Accuracy: {:.3f}'.format(cm.trace() / cm.sum()), fontsize=fs+2)
        plt.show()
        return
    else:
        return ax

def cm_plot(y_gt, y_pred, le_file_path):
    le = cluster.load_le(le_file_path)
    labels_ordered = le.inverse_transform(range(len(le.mapper)))
    CM = confusion_matrix(y_gt, y_pred)
    pretty_cm(CM, labels_ordered)
    
def pano_plot(features, dr_technique:str):
    if dr_technique == 'pca':
        pca_nw = PCA(n_components=0.95, svd_solver='full', whiten=False, random_state=42)
        reduced = pca_nw.fit_transform(features)
    elif dr_technique == 'tsne':
        reducer = umap.UMAP(n_components=2, metric='cosine', random_state=825765)
        reduced = reducer.fit_transform(features)
    X = np.array(reduced)
    tsne = TSNE(n_components=2, random_state=567).fit_transform(X)
    tx, ty = tsne[:,0], tsne[:,1]
    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))
    width = 4000
    height = 3000
    max_dim = 100
    full_image = Image.new('RGBA', (width, height))
    for img, x, y in zip(files, tx, ty):
        tile = Image.open(img)
        rs = max(1, tile.width/max_dim, tile.height/max_dim)
        tile = tile.resize((int(tile.width/rs), int(tile.height/rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width-max_dim)*x), int((height-max_dim)*y)), mask=tile.convert('RGBA'))
    plt.figure(figsize = (16,12))
    plt.imshow(full_image)
    plt.grid(None)
    return

def tsne_plot():
    return

# def gt_and_pred_plot(reduced, y_gt, y_pred, le_file_path):
#     le = cluster.load_le(le_file_path)
#     labels_ordered = le.inverse_transform(range(len(le.mapper)))
#     df = pd.DataFrame({'files': files,
#                     'x_nw':reduced[:,0],
#                     'y_nw':reduced[:,1],
#                     'labels': labels,
#                     },
#                     index=files)

#     y_pred_str = le.inverse_transform(y_pred)
#     y_gt_str = le.inverse_transform(y_gt)
#     df['y_pred_labels'] = pd.Series(y_pred_str, index=files)

#     fig, ax = plt.subplots(1,2, figsize=(8,5), dpi=150)


#     sns.scatterplot(data=df, x='x_nw', y='y_nw', hue='labels', palette='tab10', hue_order=labels_ordered, ax=ax[0]) # ground truth labels
#     sns.scatterplot(data=df, x='x_nw', y='y_nw', hue='y_pred_labels', palette='tab10', hue_order=labels_ordered, ax=ax[1]) # predicted labels

#     ax[0].get_legend().remove()
#     ax[1].legend(bbox_to_anchor=(1.05,1))
#     ax[0].set_title('ground truth labels')
#     ax[1].set_title('predicted labels')
#     fig.tight_layout()
#     plt.show()
#     return



if __name__ == "__main__":
    print("hey")
    # test data