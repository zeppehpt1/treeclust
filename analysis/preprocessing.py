import numpy as np
import skimage.exposure as skie
import glob
import cv2
import tqdm
import os

from numpy.typing import NDArray
from skimage.restoration import denoise_tv_chambolle
from pathlib import Path
from PIL import Image,ImageOps

def padding(img:NDArray, expected_size:int) -> NDArray:
    desired_size = expected_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

def resize_with_padding(img:NDArray, expected_size:int) -> NDArray:
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

def pil_resize(img:NDArray, expected_size:int) -> NDArray:
    image = img
    new_image = image.resize((expected_size,expected_size))
    return new_image

def get_png_file_names(file_dir:str) -> list:
    files = glob.glob(file_dir + '/*.png')
    return files

def normalize_img(img_arr:NDArray) -> NDArray:
    normalized_img = cv2.normalize(img_arr, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    normalized_img = normalized_img.astype(np.uint8)
    return normalized_img

def highest_pixel_count(img_array:NDArray) -> int:
    pixel, n_of_pixels = np.unique(img_array, return_counts=True)
    highest_pixel_value = pixel[np.argsort(-n_of_pixels)]
    index = 0
    forbidden_values = {0,1,2}
    while True:
        if highest_pixel_value[index] not in forbidden_values:
            return highest_pixel_value[index]
        index += 1

def check_alpha_channel(img_arr:NDArray) -> bool:
    h,w,c = img_arr.shape
    return True if c ==4 else False

def preprocess_images(images_path:str, resize:str, expected_size:int, square:bool, set_clahe:bool, set_denoising:bool, set_blur:bool) -> str:
    """Apply one or several image processing options to multiple images to prepare them for object detection or feature extraction.

    Args:
        images_path (str): Path to images
        resize (str): Realize image resizing with image 'padding' or image 'stretch'
        expected_size (int): Desired image size
        square (bool): Ensures correct file naming when square format is used
        set_clahe (bool): Enable or disable CLAHE
        set_denoising (bool): Enable or disable denoising
        set_blur (bool): Enable or disable blur

    Returns:
        str: Outputs the path to the created preprocessed image files folder.
    """
    file_list = get_png_file_names(images_path)
    shape = 'polygon'
    enhancements = ''
    if set_clahe and set_denoising:
        enhancements = '_clahe-denoising'
    elif set_clahe:
        enhancements = '_clahe'
    elif set_denoising:
        enhancements = '_denoising'
    # create dir
    out_dir = Path(images_path).parent / Path('preprocessed_' + str(expected_size) + enhancements + '_clipped_pred_' + shape + '_' + str(len(file_list)))
    if os.path.isdir(out_dir) == False:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        if square:
            shape = 'square'
        for img in (pbar := tqdm.tqdm(file_list)):
            pbar.set_description(f"Processing {img}")
            file_name = Path(img).stem
            img = cv2.imread(img) # loads image in BGR order!
            #img = plt.imread(img) # loads image in RGB order
            # remove alpha if image has alpha
            if img.shape[2] == 4:
                img = img[:,:,:3]
            # normalize brightness CLAHE
            if set_clahe:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = skie.equalize_adapthist(img)
                # normalize pixel values to range 0-255
                img = normalize_img(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # denoise image
            if set_denoising:
                img = cv2.fastNlMeansDenoisingColored(img, None, 10,10,7,21) # increases processing time!
            if set_blur:
                img = denoise_tv_chambolle(img, weight=0.1, channel_axis=-1)
            # array to img
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            PIL_image = Image.fromarray((img).astype(np.uint8)) # if normalized
            # resize
            if resize == 'padding':
                PIL_image = resize_with_padding(PIL_image,(expected_size,expected_size))
            elif resize == 'stretch':
                PIL_image = PIL_image.resize((expected_size, expected_size))
            # save img
            PIL_image.save(str(out_dir) + '/' + str(file_name) + '_preprocessed' + '.png')
        return out_dir
    else:
        print(enhancements + ' processed files already exists')
        return out_dir