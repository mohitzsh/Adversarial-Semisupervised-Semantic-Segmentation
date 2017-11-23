import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from utils.transforms import OneHotEncode

def load_image(file):
    return Image.open(file)

def read_img_list(filename):
    with open(filename) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return img_list

class PascalVOC(Dataset):

    TRAIN_LIST = "lists/train.txt"
    VAL_LIST = "lists/val.txt"

    def __init__(self, root, data_root, img_transform = Compose([]),\
     label_transform=Compose([]), co_transform=Compose([]),\
      train_phase=True):

        self.n_class = 21
        self.root = root
        self.data_root = data_root
        self.images_root = os.path.join(self.data_root, 'img')
        self.labels_root = os.path.join(self.data_root, 'cls')
        self.img_list = read_img_list(os.path.join(self.root,'datasets',self.TRAIN_LIST)) if train_phase else read_img_list(os.path.join(self.root,'datasets',self.VAL_LIST))

        self.img_transform = img_transform
        self.label_transform = label_transform
        self.co_transform = co_transform
        self.train_phase = train_phase

    def __getitem__(self, index):
        filename = self.img_list[index]

        with open(os.path.join(self.images_root,filename+'.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(os.path.join(self.labels_root,filename+'.png'), 'rb') as f:
            label = load_image(f).convert('P')

        image, label = self.co_transform((image,label))
        image = self.img_transform(image)
        label = self.label_transform(label)
        ohlabel = OneHotEncode()(label)
        # Rteurn labels as single channel tensor or as one-hot encoded tensor

        return image, label, ohlabel

    def __len__(self):
        return len(self.img_list)
