from datasets.pascalvoc import PascalVOC
from torch.utils.data import DataLoader
import generators.deeplabv2 as deeplabv2
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import numpy as np
from utils.metrics import scores
import torchvision.transforms as transforms
from utils.transforms import RandomSizedCrop, IgnoreLabelClass, ToTensorLabel, NormalizeOwn, ZeroPadding
from torchvision.transforms import ToTensor


def main():
    home_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir",help="A directory containing img (Images) \
                        and cls (GT Segmentation) folder")
    parser.add_argument("snapshot",help="Snapshot with the saved model")
    parser.add_argument("--val_orig", help="Do Inference on original size image.\
                        Otherwise, crop to 321x321 like in training ",action='store_true')
    parser.add_argument("--norm",help="Normalize the test images",\
                        action='store_true')
    args = parser.parse_args()

    if args.val_orig:
        img_transform = transforms.Compose([ToTensor()])
        if args.norm:
            img_transform = transforms.Compose([ToTensor(),NormalizeOwn(dataset='voc')])
        label_transform = transforms.Compose([IgnoreLabelClass(),ToTensorLabel()])
        co_transform = transforms.Compose([RandomSizedCrop((321,321))])

        testset = PascalVOC(home_dir,args.dataset_dir,img_transform=img_transform, \
            label_transform = label_transform,co_transform=co_transform,train_phase=False)
        testloader = DataLoader(testset,batch_size=1)
    else:
        img_transform = transforms.Compose([ZeroPadding(),ToTensor()])
        if args.norm:
            img_transform = img_transform = transforms.Compose([ZeroPadding(),ToTensor(),NormalizeOwn(dataset='voc')])
        label_transform = transforms.Compose([IgnoreLabelClass(),ToTensorLabel()])

        testset = PascalVOC(home_dir,args.dataset_dir,img_transform=img_transform, \
            label_transform=label_transform,train_phase=False)
        testloader = DataLoader(testset,batch_size=1)

    generator = deeplabv2.Res_Deeplab()
    assert(os.path.isfile(args.snapshot))
    snapshot = torch.load(args.snapshot)

    saved_net = {k.partition('module.')[2]: v for i, (k,v) in enumerate(snapshot['state_dict'].items())}
    print('Snapshot Loaded')
    generator.load_state_dict(saved_net)
    generator.eval()
    generator = nn.DataParallel(generator).cuda()
    print('Generator Loaded')
    n_classes = 21

    gts, preds = [], []

    print('Prediction Goint to Start')

    # TODO: Crop out the padding before prediction
    for img_id, (img,gt_mask) in enumerate(testloader):
        print("Generating Predictions for Image {}".format(img_id))
        gt_mask = gt_mask.numpy()[0]
        img = Variable(img.cuda())
        out_pred_map = generator(img)

        # Get hard prediction
        soft_pred = out_pred_map.data.cpu().numpy()[0]
        soft_pred = soft_pred[:,:gt_mask.shape[0],:gt_mask.shape[1]]
        hard_pred = np.argmax(soft_pred,axis=0).astype(np.uint8)
        for gt_, pred_ in zip(gt_mask, hard_pred):
            gts.append(gt_)
            preds.append(pred_)
    score, class_iou = scores(gts, preds, n_class=n_classes)

    print("Mean IoU: {}".format(score))
if __name__ == '__main__':
    main()
