from setuptools import setup, find_packages

setup(
    name='treeclust',
    version='1.0.0',
    description='Includes a automatic pipeline to evaluate the clustering capabilities of single tree crown images.',
    author='Richard Nieding',
    author_email='richard@nieding.de',
    license='MIT',
    packages=['treeclust'],
    install_requires=[
    'numpy',
    'pandas',
    'scikit-learn',
    'joblib',
    'yellowbrick',
    'geopandas',
    'rioxarray',
    'fiona',
    'rasterio',
    'tqdm',
    'scipy',
    'torch',
    'torchvision',
    'pillow',
    'shapely',
    'opencv-python',
    'matplotlib',
    'seaborn',
    'scikit-image',
    'umap-learn'
])