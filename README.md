# treeclust

<p>
<a href="https://github.com/zeppehpt1/treeclust/blob/main/LICENSE"><img alt="License: MIT" src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

Python package for automatic processing and clustering of tree species using several different image preprocessing, dimensionality reduction and clustering techniques. The current implementation works with annotated data to take advantage of tree species clustering capabilities based on high-resolution RGB image data. However, with some adjustments, the code can also be used to cluster tree species when tree species labels are not available.

The data used for training the tree detection and delineation model, as well as the data used for the analysis pipeline, can be made available upon request. Please do not hesitate to contact me if you have any questions about the data or individual parts of the process.

## Requirements

Image files (orthomosaics), corresponding shapefiles with information about the location and species of the trees, and corresponding ground truth mask image files, if applicable.

**Note**: The whole process is primarily tested under Linux, but should also work under other operating systems. You may need to use different file paths and working with conda could be different.

## Getting started

Preliminary Information:
- The ground truth mask image files used were introduced primarily because the tree species labels in the [FORTRESS](https://dx.doi.org/10.35097/538) dataset used were not appropriate for the entire pipeline due to their tree crown labels. These ground truth mask files allowed the ground truth tree species to be mapped to the new tree crown labels. Therefore, a ground truth mask image describes an image in which each pixel of the labeled tree crowns is mapped to its respective species_ID.
- For the FORTRESS dataset, "Schiefer" may sometimes be used as a synonym in the code.
- The file `dataprep.py` contains steps for data preparation (e.g. extraction of tiles, reassignment of tree type designations) to make the data compatible for the pipeline.
- The file `prepare.py` contains functions for the first pipelining step

The `__main__.py` file contains all the automatic pipelining steps. The other `__stat_driver__.py` file is used to collect the data for the statistical tests. There are two example notebooks: one shows an analysis of a single pipeline combination. The second example shows how the data and results were collected for the portability tests.

1. Extract single tree crowns (polygons) or inner square polygons from the orthomosaic
2. Preprocess images with two or more options, resulting in two or more image sets
3. Extract image features of the preprocessed image sets
4. Apply dimensionality reduction techniques on the image features
5. Cluster the reduced image features and calculate the mean value from several runs

This scheme, which starts with the extraction of single trees, visually describes the analysis steps.

![ss](imgs/whole-scheme2.png)

## Docker setup

Create a new folder for the application and create the files "docker-compose.yaml" and "conifg.env" in this folder:

```bash
mkdir treeclust
cd treeclust
touch docker-compose.yaml config.env
```

In the Docker folder of the repository there is a sample docker-compose file that downloads the latest image, searches for the configuration file and finally assigns two volumes - one for the data sets and another for the pretrained models. This ensures that the trained models are not downloaded again and again when the container is running. Now, simply copy the contents of the docker-compose file into your newly created `docker-compose.yaml`.

The configuration file defines the following parameters:
```
DATASETS_PATH=/treeclust/data # only required for execution without Docker
SITE=TRESMO # name of the dataset used
NUMBER_OF_CLASSES=10 # Number of ground truth classes in the dataset
EXPERIMENT_RUNS=5 # how often the pipeline should be executed
```

Once you have your data set available, I recommend that you create the data folder within your newly created treeclust folder. Then copy the contents of your dataset into the data folder. The resulting folder structure will look like this:

```
treeclust
│   docker-compose.yaml
│   config.env
│
└───data
│   |   
│   └TRESMO
│   │   │
│   │   └gt_masks
│   │   │   ...        
│   │   │
│   │   └ortho_tiles
│   │   │   ...
│   │   │
│   │   └pred_crown_tiles
│   │   │   ...
│   │
│   │
│   └second_dataset
```

Everything is now ready to launch the container and start the pipeline:

```bash
docker compose run
```

You can then evaluate your results by examining the final `.pickle` file in the results folder, for example with the Python package pandas.

## Non-Docker setup

Setting up folders and the environment:

```bash
mkdir treeclust
cd treeclust
git clone https://github.com/zeppehpt1/treeclust.git
conda env create --file envrionment.yaml # or use mamba
mkdir data
```

Now insert your dataset into the newly created data folder. You can then change the parameters in the `config.env` file as required.

Activate the environment and run the code:

```bash
conda activate treeclust
python treeclust/__main__.py
```

**Note**: When using conda, I strongly recommend checking and activating the libmamba solver if it is not selected.

```bash
conda config --show-sources
conda config --set solver libmamba
conda config --set solver classic # reverts to default solver
```

## TODOs

- [ ] Refactor code
- [ ] Implement tests
- [ ] Improve the separation of clusters between different deciduous tree species
- [ ] Reduce Docker image size
