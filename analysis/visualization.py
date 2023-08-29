import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import skimage.exposure as skie

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from PIL import Image
from numpy.typing import NDArray

from . import cluster
from analysis import label_tools as lt

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

def cm_plot(y_gt:list, y_pred:list, le_file_path:str):
    le = cluster.load_le(le_file_path)
    labels_ordered = le.inverse_transform(range(len(le.mapper)))
    CM = confusion_matrix(y_gt, y_pred)
    pretty_cm(CM, labels_ordered)
    
def pano_plot(features:NDArray, files:list):
    tx, ty = features[:,0], features[:,1]
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

def tsne_plot(features:NDArray, files:list, labels:list, labels_ordered:list):
    pca_nw = PCA(n_components=0.95, svd_solver='full', whiten=False, random_state=42)
    x_nw = pca_nw.fit_transform(features)
    X = np.array(x_nw)
    tsne = TSNE(n_components=2, random_state=567).fit_transform(X)
    tx, ty = tsne[:,0], tsne[:,1]

    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))
    
    labels_ordered = lt.inverse_transform(range(len(lt.mapper)))
    df = pd.DataFrame({'files': files,
                    'x_nw':tx,
                    'y_nw':ty,
                    'labels': labels,
                    },
                    index=files)

    fig, ax = plt.subplots(1,1,figsize=(6,3), dpi=150)
    sns.scatterplot(data=df, x='x_nw', y='y_nw', hue='labels', palette='tab10', hue_order=labels_ordered)
    ax.get_legend().remove()
    ax.legend(bbox_to_anchor=(1.05,1))
    ax.set_title('t-sne')
    fig.tight_layout()
    plt.show()
    return

def tsne_gt_pred_plot(features:NDArray, files:list, labels:list, y_gt:list, y_pred:list, labels_ordered:list):
    pca_nw = PCA(n_components=0.95, svd_solver='full', whiten=False, random_state=42)
    x_nw = pca_nw.fit_transform(features)
    X = np.array(x_nw)
    tsne = TSNE(n_components=2, random_state=567).fit_transform(X)
    tx, ty = tsne[:,0], tsne[:,1]

    tx = (tx-np.min(tx)) / (np.max(tx) - np.min(tx))
    ty = (ty-np.min(ty)) / (np.max(ty) - np.min(ty))
    
    df = pd.DataFrame({'files': files,
                    'x':tx,
                    'y':ty,
                    'labels': labels,
                    },
                    index=files)

    y_pred_str = lt.inverse_transform(y_pred)
    y_gt_str = lt.inverse_transform(y_gt)
    df['y_pred_labels'] = pd.Series(y_pred_str, index=files)
    df['y_pred_labels'] = pd.Series(y_pred, index=files)
    df['labels'] = pd.Series(y_gt, index=files)

    fig, ax = plt.subplots(1,2, figsize=(8,5), dpi=150)

    sns.scatterplot(data=df, x='x', y='y', hue='labels', palette='tab10', hue_order=sorted(set(y_gt)), ax=ax[0]) # ground truth labels
    sns.scatterplot(data=df, x='x', y='y', hue='y_pred_labels', palette='tab10', hue_order=sorted(set(y_gt)), ax=ax[1]) # predicted labels

    ax[0].get_legend().remove()
    ax[1].legend(bbox_to_anchor=(1.05,1))
    ax[0].set_title('ground truth labels')
    ax[1].set_title('predicted labels')
    ax[0].set_ylabel(None)
    ax[1].set_ylabel(None)
    ax[0].set_xlabel(None)
    ax[1].set_xlabel(None)
    fig.tight_layout()
    plt.show()

def clahe_plot(img_filepath:str):
    img = cv2.imread(img_filepath)
    fig, (ax0) = plt.subplots(ncols=1, sharex=True, sharey=True)
    img0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img0 = ax0.imshow(skie.equalize_adapthist(img0))
    ax0.set_title("")
    ax0.axis("off")
    return

def clahe_denoising_plot(img_filepath:str):
    img = cv2.imread(img_filepath)
    fig, (ax0) = plt.subplots(ncols=1, sharex=True, sharey=True)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2 = skie.equalize_adapthist(img2)
    img2 = cv2.normalize(img2, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    img2 = img2.astype(np.uint8)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    img2 = cv2.fastNlMeansDenoisingColored(img2, None,10,10,7,21)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = ax0.imshow(img2, cmap=plt.cm.gray)
    ax0.set_title("")
    ax0.axis("off")
    return

if __name__ == "__main__":
    print("hey")
    # test data