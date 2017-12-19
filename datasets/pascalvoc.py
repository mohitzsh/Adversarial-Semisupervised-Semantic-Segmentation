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
    return np.array(img_list)

class PascalVOC(Dataset):

    TRAIN_LIST = "lists/train.txt"
    VAL_LIST = "lists/val.txt"

    def __init__(self, root, data_root, img_transform = Compose([]),\
     label_transform=Compose([]), co_transform=Compose([]),\
      train_phase=True,split=1,labeled=True,seed=0):
        np.random.seed(100)
        self.n_class = 21
        self.root = root
        self.data_root = data_root
        self.images_root = os.path.join(self.data_root, 'img')
        self.labels_root = os.path.join(self.data_root, 'cls')
        self.img_list = read_img_list(os.path.join(self.root,'datasets',self.TRAIN_LIST)) \
                        if train_phase else read_img_list(os.path.join(self.root,'datasets',self.VAL_LIST))
        self.split = split
        self.labeled = labeled
        n_images = len(self.img_list)
        self.img_l = np.random.choice(range(n_images),int(n_images*split),replace=False) # Labeled Images
        self.img_u = np.array([idx for idx in range(n_images) if idx not in self.img_l],dtype=int) # Unlabeled Images
        if self.labeled:
            self.img_list = self.img_list[self.img_l]
        else:
            self.img_list = self.img_list[self.img_u]
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

        return image, label, ohlabel

    def __len__(self):
        return len(self.img_list)
