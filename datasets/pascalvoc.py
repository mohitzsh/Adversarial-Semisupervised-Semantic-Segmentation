import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

def load_image(file):
    return Image.open(file)

def read_img_list(filename):
    with open(filename) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return img_list


"""
    Implicitly applies ToTensor() transformation to both image and label
"""
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

        # # TODO: Add this as a transform for the label
        # label = Image.eval(label,lambda p: 0 if p == 255 else p)
        #
        #
        # #If the test phase, pad the image and label with zeros to make the size 513x513
        # # if not self.train_phase:
        # if False:
        #     # New image is 513x513x3
        #     img_new = np.zeros((513,513,3),np.uint8)
        #     img_orig = np.array(image,np.uint8)
        #     img_new[:img_orig.shape[0],:img_orig.shape[1],:] = img_orig
        #     image = img_new
        #
        #     # No Padding for target mask.
        #     label = np.array(label,dtype=np.uint8)
        #
        #     # Convert to Tensors
        #     label = torch.from_numpy(label).long()
        #     image = transforms.ToTensor()(image)
        # else:
        #     # image = transforms.ToTensor()(image)
        #     # In training, label still needs to be converted to tensor
        #     label_a = np.array(label,dtype=np.uint8)
        #     label = torch.from_numpy(label_a).long()

        return image, label

    def __len__(self):
        return len(self.img_list)
