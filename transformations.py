import numpy as np
import torch
from scipy.ndimage import zoom
from skimage.transform import resize

def normalise(image):
    image = (image - image.min()) / (image.max() - image.min())
    image = image.astype(np.float32)
    return image

def resize_image(image):
    new_depth  = 40
    new_width = 256
    new_height = 256
    image = resize(image, (new_depth, new_width, new_height),anti_aliasing=True , order=5,mode='constant')
    return image

def np_to_tensor(image):
    image = torch.from_numpy(image)
    return image

