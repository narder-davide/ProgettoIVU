from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
import random

import io
import os
import pandas as pd

from torch.utils.data import Dataset
import torch


class CardsDataset(Dataset):
    def __init__(self, txt_path='filelist.txt', img_dir='data', transform=None):
        """
        Initialize data set as a list of IDs corresponding to each item of data set

        :param img_dir: path to image files as a uncompressed tar archive
        :param txt_path: a text file containing names of all of images line by line
        :param transform: apply some transforms like cropping, rotating, etc on input image
        """

        df = pd.read_csv(txt_path, sep=' ', index_col=0)
        self.img_names = df.index.values
        self.txt_path = txt_path
        self.img_dir = img_dir
        self.transform = transform
        self.to_tensor = ToTensor()
        self.to_pil = ToPILImage()


    def get_image_from_folder(self, name):
        """
        gets a image by a name gathered from file list text file

        :param name: name of targeted image
        :return: a PIL image
        """

        image = Image.open(os.path.join(self.img_dir, name))
        return image

    def __len__(self):
        """
        Return the length of data set using list of IDs

        :return: number of samples in data set
        """
        return len(self.img_names)

    def __getitem__(self, index):
        """
        Generate one item of data set.

        :param index: index of item in IDs list

        :return: a sample of data as a dict
        """

    
        X = self.get_image_from_folder(self.img_names[index])

        # Get you label here using available pandas functions
        Y = "" #########

        if self.transform is not None:
            X = self.transform(X)
#            Y = self.transform(Y) # if your label is image too - remove if it is number

        sample = {'X': X,
                  'Y': Y}

        return sample