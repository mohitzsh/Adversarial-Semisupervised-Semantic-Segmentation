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

def val(model,valoader,nclass=21):
    model.eval()
    gts, preds = [], []
    for img_id, (img,gt_mask) in enumerate(valoader):
        gt_mask = gt_mask.numpy()[0]
        img = Variable(img.cuda(),volatile=True)
        out_pred_map = model(img)

        # Get hard prediction
        soft_pred = out_pred_map.data.cpu().numpy()[0]
        soft_pred = soft_pred[:,:gt_mask.shape[0],:gt_mask.shape[1]]
        hard_pred = np.argmax(soft_pred,axis=0).astype(np.uint8)
        for gt_, pred_ in zip(gt_mask, hard_pred):
            gts.append(gt_)
            preds.append(pred_)
    miou, _ = scores(gts, preds, n_class = nclass)

    return miou
