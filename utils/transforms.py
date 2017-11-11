import torch
from PIL import Image
import numpy as np
import math
import random
import torchvision.transforms as transforms

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

            # w_label = int(round(math.sqrt(target_label_area * rand_aspect_ratio)))
            # h_label = int(round(math.sqrt(target_label_area / rand_aspect_ratio)))

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
        # Add a fallback method
        img_scale = transforms.Scale(self.size,interpolation=self.img_interpolation)
        label_scale = transforms.Scale(self.size,interpolation=self.label_interpolation)
        crop = transforms.CenterCrop(self.size)
        return crop(img_scale(img)), crop(label_scale(label_scale))
