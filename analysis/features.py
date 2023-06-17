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
from glob import glob
from tqdm import tqdm

from analysis import label_tools as lt
from .constants import SITE

def load_files(preprocessed_fp:str):
    files = sorted(Path(preprocessed_fp).glob('*.png'))
    randomizer = np.random.RandomState(seed=12)
    randomizer.shuffle(files)
    return files

def alter_image(img_name, image_size):
    resize = transforms.Resize((image_size,image_size))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    image = Image.open(img_name)
    image = resize(image)
    image_tensor = normalize(to_tensor(image)).unsqueeze(0)
    image_tensor = image_tensor.reshape(1,3,image_size,image_size)
    return image_tensor

def load_images_as_tensors(files, image_size):
    image_tensors = [alter_image(image, image_size) for image in files]
    return image_tensors

def extract_numeric_labels(files):
    return [filename.stem.split('_')[6] for filename in files]
# TODO: use consistent position for the ID
# mabye adjust number position in filename when attribute error!
# 6 schiefer
# 5 stadtwald

# # single cfb184 crowns
# def convert_number_to_str(labels):
#     update = {
#     '4':'Fagus_sylvatica',
#     #'5':'Fraxinus_excelsior',
#     #'6':'Quercus_spec',
#     '8':'deadwood',
#     '10':'Abies_alba',
#     #'11':'Larix_decidua',
#     '12':'Picea_abies',
#     #'13':'Pinus_sylvestris',
#     #'14':'Pseudotsuga_menziesii'
#     }
#     updated_labels = (pd.Series(labels)).map(update)
#     species_labels = list(updated_labels)
#     return species_labels
# # adjust species according to analyzed sites

def convert_number_to_str(labels): # all schiefer
#TODO only one scheme for all datasets, it doesn't matter if one number is not present in the dataset
    if SITE == 'Schiefer':
        update = {
        #'2':'Acer pseudoplatanus', not appears
        '4':'Fagus_sylvatica',
        '5':'Fraxinus_excelsior',
        '6':'Quercus_spec',
        '8':'deadwood',
        '10':'Abies_alba',
        '11':'Larix_decidua',
        '12':'Picea_abies',
        '13':'Pinus_sylvestris',
        '14':'Pseudotsuga_menziesii',
        '15':'Betula_pendula'
        }
        updated_labels = (pd.Series(labels)).map(update)
        species_labels = list(updated_labels)
        return species_labels
    elif SITE == 'Bamberg_Stadtwald':
        update = {
        '2':'Acer pseudoplatanus',
        '4':'Fagus_sylvatica',
        '5':'Fraxinus_excelsior',
        '6':'Quercus_spec',
        '8':'deadwood',
        '10':'Abies_alba',
        '11':'Larix_decidua',
        '12':'Picea_abies',
        '13':'Pinus_sylvestris',
        '14':'Pseudotsuga_menziesii',
        '15':'Betula pendula',
        '16':'Tilia',
        '18':'Lbh',
        '19':'Sorbus_torminalis',
        '20':'Ulmus',
        '21':'Acer_platanoides',
        '22':'Quercus_rubra'
        }
        updated_labels = (pd.Series(labels)).map(update)
        species_labels = list(updated_labels)
        return species_labels
    elif SITE == 'Tretzendorf':
        update = {
        '4':'Fagus_sylvatica',
        '6':'Quercus_spec',
        '8':'deadwood',
        '11':'Larix_decidua',
        '12':'Picea_abies',
        '13':'Pinus_sylvestris',
        '18':'Lbh',
        '19':'Sorbus_torminalis',
        '20':'Ulmus',
        '22':'Quercus_rubra'
        }
        updated_labels = (pd.Series(labels)).map(update)
        species_labels = list(updated_labels)
        return species_labels
# adjust species according to analyzed sites

def encode_labels(labels):
    le = lt.CustomLabelEncoder()
    le.fit(labels, sorter=lambda x: x.upper())
    return le

def save_le(le, le_path):
    with open(le_path, 'wb') as f:
        pickle.dump(le, f)
        

#________________________________________

# define fe model for vgg16
class FeatureExtractor(nn.Module):
  def __init__(self, model):
    super(FeatureExtractor, self).__init__()
		# Extract VGG-16 Feature Layers
    self.features = list(model.features)
    self.features = nn.Sequential(*self.features)
		# Extract VGG-16 Average Pooling Layer
    self.pooling = model.avgpool
		# Convert the image into one-dimensional vector
    self.flatten = nn.Flatten()
		# Extract the first part of fully-connected layer from VGG16
    self.fc = model.classifier[0]
  
  def forward(self, x):
		# It will take the input 'x' until it returns the feature vector called 'out'
    out = self.features(x)
    out = self.pooling(out)
    out = self.flatten(out)
    out = self.fc(out) 
    return out 

def get_model(cnn: str):
    if cnn == 'vgg':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model = FeatureExtractor(model)
    elif cnn == 'resnet':
        model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
    elif cnn == 'effnet':
        model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
    elif cnn == 'densenet':
        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
    elif cnn == 'inception':
        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    model.eval()
    return model

def concat_tensors(features):
    fc = torch.cat(features)
    fc = fc.cpu().detach().numpy()
    return fc

def get_avgpool_fc_feature(model, image_tensor):
    # a dict to store the activations
    activation = {}
    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    # register forward hooks on the layers of choice
    h1 = model.avgpool.register_forward_hook(getActivation('avgpool'))
    # forward pass -- getting the outputs
    out = model(image_tensor)
    # detach the hooks
    h1.remove()
    feature = torch.squeeze(activation['avgpool'])
    feature = torch.unsqueeze(feature,dim=0)
    return feature

def get_avgpool_fc_features(model, image_tensors):
    features = [get_avgpool_fc_feature(model, tensor) for tensor in tqdm(image_tensors)]
    features = concat_tensors(features)
    return features

def get_avg_last_fc_feature(model, image_tensor):
     # a dict to store the activations
    activation = {}
    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    # register forward hooks on the layers of choice
    h1 = model.register_forward_hook(getActivation('avg_last'))
    # forward pass -- getting the outputs
    out = model(image_tensor)
    # detach the hooks
    h1.remove()
    feature = torch.squeeze(activation['avg_last'])
    feature = torch.unsqueeze(feature,dim=0)
    return feature

def get_avg_last_fc_features(model, image_tensors):
    features = [get_avg_last_fc_feature(model, tensor) for tensor in tqdm(image_tensors)]
    features = concat_tensors(features)
    return features

def get_fc1_feature(model, image_tensor):
    with torch.no_grad():
        feature = model(image_tensor)
    feature.cpu().detach().numpy()
    return feature

def get_fc1_features(model, image_tensors):
    with torch.no_grad():
        features = [model(tensor) for tensor in tqdm(image_tensors)]
    features = concat_tensors(features)
    return features

def extract_encodings(cnn:str, files:str):
    if cnn == 'vgg':
        image_size = 224
        model = get_model(cnn)
        tensors = load_images_as_tensors(files, image_size)
        print("Get VGG16 encodings")
        features = get_fc1_features(model, tensors)
    elif cnn == 'resnet':
        image_size = 224
        model = get_model(cnn)
        tensors = load_images_as_tensors(files, image_size)
        print("Get ResNet152 encodings")
        features = get_avgpool_fc_features(model, tensors)
    elif cnn == 'effnet':
        image_size = 224
        model = get_model(cnn)
        tensors = load_images_as_tensors(files, image_size)
        print("Get EffNetV2 encodings")
        features = get_avgpool_fc_features(model, tensors)
    elif cnn == 'densenet':
        image_size = 299
        model = get_model(cnn)
        tensors = load_images_as_tensors(files, image_size)
        print("Get DenseNet201 encodings")
        features = get_avg_last_fc_features(model, tensors)
    elif cnn == 'inception':
        image_size = 224
        model = get_model(cnn)
        tensors = load_images_as_tensors(files, image_size)
        print("Get InceptionV3 encodings")
        features = get_avgpool_fc_features(model, tensors)
    return features

def save_encodings(files, features, labels, features_path):
    results = {'filename': files,
           'features': features,
           'labels': labels,
           'layer_name': 'fc'}
    with open(features_path, 'wb') as f:
        pickle.dump(results, f)
        
def create_and_save_le_encodings(cnn:str, preprocessed_fp:str, site_folder:str):
    
    features_dir = site_folder + 'encodings/'
    Path(features_dir).mkdir(parents=True, exist_ok=True)
    preprocess_str = str(Path(preprocessed_fp).stem).split('_')[2]
    features_path = features_dir + cnn + '_' + preprocess_str + '_polygon_pred.pickle'

    if os.path.isfile(features_path) == False:
        files = load_files(preprocessed_fp)
        labels = extract_numeric_labels(files)
        print(labels)
        labels = convert_number_to_str(labels)
        
        le = encode_labels(labels)
        features = extract_encodings(cnn, files)
        
        # save le
        le_dir = site_folder + '/label_encodings/'
        Path(le_dir).mkdir(parents=True, exist_ok=True)
        le_path = le_dir + cnn + '_' + preprocess_str + '_label_encodings.pickle'
        save_le(le, le_path)
        
        # save encoding
        save_encodings(files, features, labels, features_path)
        return le_path, features_path
    return

if __name__ == "__main__":
    print("write a test case or something")