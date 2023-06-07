import torch
import os
# from skimage.io import imread
import SimpleITK as sitk
from torch.utils import data
import numpy as np


class SegmentationDataSet(data.Dataset):
    def __init__(self,
                 data_dir,
                 train=True,
                 transform=None
                 ):
        self.train = train
        if self.train:
            self.subdir = 'train'
        else:
            self.subdir = 'test'
        self.image_dir = os.path.join(data_dir, self.subdir, 'images')
        self.target_dir = os.path.join(data_dir, self.subdir, 'images')
        self.transform = transform
        # self.inputs_dtype = torch.float32
        # self.targets_dtype = torch.long
        self.image_list = os.listdir(self.image_dir)
        self.target_list = os.listdir(self.target_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self,
                    index: int):
        # Select the sample
        input_loc = os.path.join(self.image_dir, self.image_list[index])
        target_loc = os.path.join(self.target_dir, self.target_list[index])

        # Load input and target
        sitk.ProcessObject_SetGlobalWarningDisplay(False)
        x, y = sitk.ReadImage(input_loc), sitk.ReadImage(target_loc)
        x, y = sitk.GetArrayFromImage(x), sitk.GetArrayFromImage(y)
        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x), self.transform(y)

        # Typecasting
        # x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y
