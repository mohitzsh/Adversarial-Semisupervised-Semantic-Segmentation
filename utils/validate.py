from datasets.pascalvoc import PascalVOC
from torch.utils.data import DataLoader
import generators.deeplabv2 as deeplabv2
import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import numpy as np
from utils.metrics import scores
import torchvision.transforms as transforms
from utils.transforms import RandomSizedCrop, IgnoreLabelClass, ToTensorLabel, NormalizeOwn, ZeroPadding
from torchvision.transforms import ToTensor,Compose

class Validator(object):
    """
        Evaluate the generator network on valloader.
    """
    def __init__(self,model,valoader):

        self._model = model # Is it risky????
        self._valoader = valoader
        self._nclass = 21

    def validate(self):

        gts, preds = [], []
        for img_id, (img,gt_mask) in enumerate(self._valoader):
            gt_mask = gt_mask.numpy()[0]
            img = Variable(img.cuda())
            out_pred_map = self._model(img)

            # Get hard prediction
            soft_pred = out_pred_map.data.cpu().numpy()[0]
            soft_pred = soft_pred[:,:gt_mask.shape[0],:gt_mask.shape[1]]
            hard_pred = np.argmax(soft_pred,axis=0).astype(np.uint8)
            for gt_, pred_ in zip(gt_mask, hard_pred):
                gts.append(gt_)
                preds.append(pred_)
        miou, _ = scores(gts, preds, n_class=self._nclass)

        return miou
