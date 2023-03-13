import numpy as np

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import skimage

class CustomLabelEncoder:
    """
    Creates a mapping between string labels and integer class clabels for working with categorical data.
    
    
    Attributes
    ----------
    mapper:None dict
        None if mapper is not supplied or model is not fit.
        keys are unique string labels, values are integer class labels.
    """
    def __init__(self, mapper=None):
        """
        Initializes class instance.
        
        If the mapper dictionary is supplied here, then the model can be used without calling .fit().
        
        Parameters
        -----------
        mapper (optional): dict or None
            if mapper is None encoder will need to be fit to data before it can be used.
            If it is a dictionary mapping string labels to integer class labels, then this will be stored
            and the model can be used to transform data.
        """
        self.mapper = mapper
    
    def fit(self, str_labels, sorter=None):
        """
        Fits string labels to intiger indices with optional sorting.
        
        np.unique() is used to extract the unique values form labels. If 
        
        Parameters
        ----------
        str_labels: list-like
            list or array containing string labels
        
        sorter (optional): None or function
            key for calling sorted() on data to determine ordering of the numeric indices for each label.
            
        Attributes
        -----------
        mapper: dict
            dictionary mapping string labels to the sorted integer indices is stored after fitting.
        
        """
        sorted_unique = sorted(np.unique(str_labels), key=sorter)
        mapper = {label: i for i, label in enumerate(sorted_unique)}
        self.mapper = mapper    

    def transform(self, str_labels):
        """
        Maps string labels to integer labels.
        
        Parameters
        ----------
        str_labels: list-like
            list of string labels whose elements are in self.mapper
        
        Returns
        --------
        int_labels: array
            array of integer labels  corresponding to the string labels
        """
        assert self.mapper is not None, 'Encoder not fit yet!'
        
        int_labels = np.asarray([self.mapper[x] for x in str_labels], np.int)
        
        return int_labels
        
    def inverse_transform(self, int_labels):
        """
        Maps integer labels to original string labels.
        
        Parameters
        -----------
        int_labels: list-like
            list or array of integer class indices
        
        Returns
        ----------
        str_labels: array(str)
            array of string labels corresponding to intiger indices
        
        """
        assert self.mapper is not None, 'Encoder not fit yet!'
        
        reverse_mapper = {y:x for x,y in self.mapper.items()}
        
        str_labels = np.asarray([reverse_mapper[x] for x in int_labels])
        
        return str_labels
    
    @property
    def labels_ordered(self):
        """
        Returns an array containing the string labels in order of which they are stored.
        
        For example, if the label_encoder has the following encoding: {'a':1,'c':3,'b':2},
        then this will return array(['a','b','c'])
        """
        pass
    
    @labels_ordered.getter
    def labels_ordered(self):
        return self.inverse_transform(range(len(self.mapper)))
    
def label_matcher(y_cluster, labels, return_mapper=False):
    """
    maps cluster centers to true labels based on the most common filename for each cluster. 
    Parameters
    ----------
    y_cluster: ndarray
        n-element array of labels obtained from clusters
        
    labels: ndarray
        n-element array of ground truth labels for which y_cluster will be mapped to
        
    return_mapper:bool
        if True, dictionary mapping values in y_cluster to values in labels will be returned
    Returns
    -----------
    y_pred: ndarray
        n-element array of values in y_cluster mapped to labels
    
    mapper (optional): dict
        dictonary whose keys are elements of y_cluster and values are the corresponding
        elements of labels.
    """
        
    y_cluster = np.asarray(y_cluster)
    labels = np.asarray(labels)
    
    y_cluster_unique = np.unique(y_cluster)

    
    mapper = {}  # keys will be cluster ID's, values will be corresponding label
    
    for x in y_cluster_unique:
        unique, counts = np.unique(labels[y_cluster==x], return_counts=True)  # get frequency of each gt label in cluster x
        mapper[x] = unique[counts.argmax()]  # set mapper[x] to the most frequent label in the cluster

    y_pred = np.asarray([mapper[x] for x in y_cluster])  # map cluster id's to labels

    if return_mapper:
        return y_pred, mapper
    else:
        return y_pred

def pretty_cm(cm, labelnames, cscale=0.6, ax0=None, fs=6, cmap='cool'):
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
    ax.set_xticklabels(labels=labelnames, minor=True, fontsize=fs)
    ax.set_yticklabels(labels=reversed(labelnames), minor=True, fontsize=fs)

    ax.set_xlabel('Predicted Labels', fontsize=fs)
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
    
def pano_plot(x, y, paths, patch_size=(3, 3), ax0=None):
    """
    Graphs y vs x with images on plot instead of points.
    Generates 'panoramic' image plots which are useful for visualizing how images 
    separate in feature space for clustering and classification challenges.
    
    Parameters
    ---------------
    x, y: ndarray
        n-element arrays of x and y coordinates for plot
        
    paths: list of strings or path objects
        n-element list of paths to images to be displaied at each point
        
    patch_size: tuple(int, int)
        size of the image patches displayed at each point
        
    ax0: None or matplotlib axis object
        if None, a new figure and axis will be created and the visualization will be displayed.
        if an axis is supplied, the panoramic visualization will be plotted on the axis in place.
        
    Returns
    ----------
    None
    
    """
    if ax0 is None:
        fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    else:
        ax = ax0
    px, py = patch_size
    ax.scatter(x, y, color=(0, 0, 0, 0))
    for xi, yi, pi in zip(x, y, paths):
        im = skimage.io.imread(pi)
        ax.imshow(im, extent=(xi - px, xi + px, yi - py, yi + py), cmap='gray')

    if ax0 is None:
        plt.show()
