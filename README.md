# treeclust

Python package for automatic processing and clustering of tree species using several different image preprocessing, dimensionality reduction and clustering techniques. The current implementation works with annotated data to take advantage of tree species clustering capabilities based on high-resolution RGB image data. However, with some adjustments, the code can also be used to cluster tree species when tree species labels are not available.

The data used for training the tree detection and delineation model, as well as the data used for the analysis pipeline, can be made available upon request.

## Requirements

- Python 3.8+
- gdal
- sklearn
- geopandas
- numpy
- fcmeans
- rioxarray
- fiona
- rasterio
- geocube
- shapely
- Image files (orthomosaics), corresponding shapefiles with information about the location and species of the trees, and corresponding ground truth mask image files, if applicable.

## Getting started

Preliminary Information:
- The ground truth mask image files used were introduced primarily because the tree species labels in the [FORTRESS](https://dx.doi.org/10.35097/538) dataset used were not appropriate for the entire pipeline due to their tree crown labels. These files allowed the ground truth tree species to be mapped to the new tree canopy labels. A ground truth mask image describes an image in which each pixel of the labeled tree crowns is mapped to its respective species_ID.
- For the FORTRESS dataset, "Schiefer" is often used as a synonym in the code.
- The file `dataprep.py` contains steps for data preparation (e.g. extracting tiles, remapping tree species labels)
- The file `prepare.py` contains functions for the first pipelining step

The main driver file contains all the automatic pipelining steps. The other stat driver file is used to collect the data for the statistical tests. There are two example notebooks: one shows an analysis of a single pipeline combination. The second example shows how the data and results were collected for the portability tests. The file `constants.py` defines global parameters regarding the analyzed forest area and how often the experiment should be run to calculate the mean values.

1. Extract single tree crowns (polygons) or inner square polygons from the orthomosaic
2. Preprocess images with two or more options, resulting in two or more image sets
3. Extract image features of the preprocessed image sets
4. Apply dimensionality reduction techniques on the image features
5. Cluster the reduced image features

This scheme, which starts with the extraction of single trees, visually describes the analysis steps.

![ss](imgs/whole-scheme2.png)

