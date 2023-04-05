import torch
import numpy as np
from numpy import asarray
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import permute
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
import pickle
import pandas as pd
from tqdm import tqdm

import label_tools as lt

def alter_image(img_name, image_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    image = Image.open(img_name)
    image_tensor = normalize(to_tensor(image)).unsqueeze(0)
    image_tensor = image_tensor.reshape(1,3,image_size,image_size)
    return image_tensor

def load_images_as_tensors(filepaths, image_size):
    image_tensors = [alter_image(image, image_size) for image in filepaths]
    return image_tensors

def extract_numeric_labels(filepaths): return [filename.stem.split('_')[4] for filename in filepaths]

def convert_number_to_str(labels):
    update = {
    '4':'Fagus_sylvatica',
    '5':'Fraxinus_excelsior',
    '6':'Quercus_spec',
    '8':'deadwood',
    '10':'Abies_alba',
    '11':'Larix_decidua',
    '12':'Picea_abies',
    '13':'Pinus_sylvestris',
    '14':'Pseudotsuga_menziesii'
    }
    updated_labels = (pd.Series(labels)).map(update)
    species_labels = list(updated_labels)
    return species_labels
