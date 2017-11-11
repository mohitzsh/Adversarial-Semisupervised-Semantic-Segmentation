import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

def load_image(file):
    return Image.open(file)

def image_path(root, basename, extension):
    return os.path.join(root,basename+extension)

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def read_img_list(filename):
    with open(filename) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return img_list[:50]

def extract_class_mask(label,c):
# Get one-hot encoding
    if c == 0:
        encoded_label = Image.eval(label,lambda p: 0 if (p != 0 or p != 255) else 1)
    else:
        encoded_label = Image.eval(label,lambda p: 0 if p != c else 1)

    return encoded_label

class PascalVOC(Dataset):

    TRAIN_LIST = "lists/train.txt"
    VAL_LIST = "lists/val.txt"

    def __init__(self, root, data_root, transform = None, co_transform=None, train_phase=True,numClasses=20):
        self.root = root
        self.data_root = data_root
        self.images_root = os.path.join(self.data_root, 'img')
        self.labels_root = os.path.join(self.data_root, 'cls')
        self.img_list = read_img_list(os.path.join(self.root,'datasets',self.TRAIN_LIST)) if train_phase else read_img_list(os.path.join(self.root,'datasets',self.VAL_LIST))

        self.transform = transform
        self.co_transform = co_transform

    def __getitem__(self, index):
        filename = self.img_list[index]

        with open(os.path.join(self.images_root,filename+'.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(os.path.join(self.labels_root,filename+'.png'), 'rb') as f:
            label = load_image(f).convert('P')

        # Apply Random Crop and resize to both label and image
        image, label = self.co_transform((image,label))
        # TODO: Add this as a transform for the label
        label = Image.eval(label,lambda p: 0 if p == 255 else p)
        image = self.transform(image)
        label = torch.from_numpy(np.array(label.getdata()).reshape(label.size))
        return image, label

    def __len__(self):
        return len(self.img_list)
