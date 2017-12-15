import torch
from PIL import Image
import numpy as np
import math
import random
import torchvision.transforms as transforms

stats = {
    'voc': {
        'mean': np.array([0.485,0.456,0.406],float),
        'std': np.array([0.229,0.224,0.225],float)
    }
}

class OneHotEncode(object):
    """
        Takes a Tensor of size 1xHxW and create one-hot encoding of size nclassxHxW
    """
    def __init__(self,nclass=21):
        self.nclass = nclass

    def __call__(self,label):
        label_a = np.array(transforms.ToPILImage()(label.byte().unsqueeze(0)),np.uint8)

        ohlabel = np.zeros((self.nclass,label_a.shape[0],label_a.shape[1])).astype(np.uint8)

        for c in range(self.nclass):
            ohlabel[c:,:,:] = (label_a == c).astype(np.uint8)

        # # Do Some assertion
        # print("Assertion about to be made")
        # for c in range(self.nclass):
        #     for i in range(321):
        #         for j in range(321):
        #             if ohlabel[c][i][j] == 1:
        #                 assert(label_a[i][j] == c)

        return torch.from_numpy(ohlabel)

class NormalizeOwn(object):
    """
        Normalize the dataset to zero mean and unit standard deviation.
    """
    def __init__(self,dataset='voc'):
        self.dataset = dataset

    def __call__(self,img):
        return transforms.Normalize(mean=stats[self.dataset]['mean'],std=stats[self.dataset]['std'])(img)

class IgnoreLabelClass(object):
    """
        Convert a label for a class to be ignored to some other class
    """
    def __init__(self,ignore=255,base=0):
        self.ignore = ignore
        self.base = base

    def __call__(self,label):
        return Image.eval(label,lambda p: self.base if p == self.ignore else p)

class ToTensorLabel(object):
    """
        Take a Label as PIL.Image with 'P' mode and convert to Tensor
    """
    def __init__(self,tensor_type=torch.LongTensor):
        self.tensor_type = tensor_type

    def __call__(self,label):
        label = np.array(label,dtype=np.uint8)
        label = torch.from_numpy(label).type(self.tensor_type)

        return label

class ZeroPadding(object):
    """
        Add zero padding to the image to right and bottom to resize it.
        Needed at test phase to make all images 513x513.

        Input: PIL Image with 'RGB' mode
        Output: Zero padded PIL image with agin with 'RGB' mode

    """
    def __init__(self,size=(513,513)):
        self.size = size


    def __call__(self,img):
        assert(img.size[0]<=self.size[0] and img.size[1] <= self.size[1])

        img_new = np.zeros((self.size[0],self.size[1],3),np.uint8)
        img_orig = np.array(img,np.uint8)
        img_new[:img_orig.shape[0],:img_orig.shape[1],:] = img_orig
        return img_new

class RandomSizedCrop(object):
    """
        RandomSizedCrop for both the image and the label
    """
    def __init__(self,size,img_interpolation=Image.BILINEAR,label_interpolation=Image.NEAREST):
        self.size = size
        self.img_interpolation = img_interpolation
        self.label_interpolation = label_interpolation

    """
        Apply the random resized crop to both (img,label)
        Expects img,label to be PIL.Image objects
    """
    def __call__(self,data):
        img = data[0]
        label = data[1]
        for attempt in range(10):
            rand_scale = random.uniform(0.08,1.0)
            rand_aspect_ratio = random.uniform(3. / 4, 4. / 3)

            area = img.size[0]*img.size[1]
            target_area = rand_scale*area

            w = int(round(math.sqrt(target_area * rand_aspect_ratio)))
            h = int(round(math.sqrt(target_area / rand_aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                label = label.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))
                assert(label.size == (w,h))

                return img.resize(self.size, self.img_interpolation),label.resize(self.size,self.label_interpolation)
        #Add a fallback method
        img_scale = transforms.Scale(self.size[0],interpolation=self.img_interpolation)
        label_scale = transforms.Scale(self.size[0],interpolation=self.label_interpolation)
        crop = transforms.CenterCrop(self.size)
        return crop(img_scale(img)), crop(label_scale(label))
