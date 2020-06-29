from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import os

from torch.utils.data import Dataset


class CardsDataset(Dataset):
    def __init__(self, img_dir='data', train=True, transform=None):

        """
        Initialize data set as a list of IDs corresponding to each item of data set

        :param img_dir: path to image files as a uncompressed tar archive
        :param txt_path: a text file containing names of all of images line by line
        :param transform: apply some transforms like cropping, rotating, etc on input image
        """

        self.seeds = ["bastoni", "spade", "coppe", "denari"]
        self.labels={}
        self.n_cards = 20
        self.img_names = []
        self.train=train

        self.img_dir = img_dir
        self.transform = transform
        self.to_tensor = ToTensor()
        self.to_pil = ToPILImage()

        #carica carte
        for seed in self.seeds:
            for j in range(1,11):
                for i in range(0,self.n_cards):
                    str=os.path.join(self.img_dir,"{}_{}/{}_{}{}.png".format(j, seed, j, seed, i))
                    if os.path.isfile(str):
                        self.img_names.append(str)
                        self.labels[len(self.img_names)-1]=self.seeds.index(seed)*10+j-1

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        """
        Generate one item of data set.

        :param index: index of item in IDs list

        :return: a sample of data as a dict
        """

        X = Image.open(self.img_names[index])

        if self.transform is not None:
            X = self.transform(X)

        return X,self.index_to_label(index)

    def index_to_label(self,idx):
        return self.labels[idx]

    def index_to_string(self, idx):
        return self.img_names[idx].split("/")[1]
