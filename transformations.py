import numpy as np
import torch
from scipy.ndimage import zoom
from skimage.transform import resize

def normalise(image):
    image = (image - image.min()) / (image.max() - image.min())
    image = image.astype(np.float32)
    return image

def resize_image(image):
    new_depth = 40
    new_width = 256
    new_height = 256
    image = resize(image, (new_depth, new_width, new_height),anti_aliasing=True , order=5,mode='constant')
    return image

def reshape_mask(mask):
    # Specify the desired new shape
    new_depth = 40
    new_width = 256
    new_height = 256

    # Reshape the mask using zoom and nearest neighbor interpolation
    resized_mask = zoom(mask, (new_depth / mask.shape[0], new_width / mask.shape[1], new_height / mask.shape[2]), order=0)

    # Convert the resized_mask to integers to remove any interpolated values
    resized_mask = resized_mask.astype(np.int32)
    return resized_mask

def np_to_tensor(image):
    image = torch.from_numpy(image)
    return image

