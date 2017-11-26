from __future__ import unicode_literals

from collections import OrderedDict
import torch
from datasets.pascalvoc import PascalVOC
import generators.deeplabv2 as deeplabv2
import discriminators.discriminator as dis
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.transforms import RandomSizedCrop, IgnoreLabelClass, ToTensorLabel, NormalizeOwn,ZeroPadding, OneHotEncode
from utils.lr_scheduling import poly_lr_scheduler
import torch.nn.functional as F
import torch.nn as nn
from functools import reduce
import torch.optim as optim
import os
import argparse
from torchvision.transforms import ToTensor,Compose
from utils.validate import val
from utils.helpers import pascal_palette_invert
import torchvision.transforms as transforms
import PIL.Image as Image
from discriminators.discriminator import Dis
def main():

    home_dir = os.path.dirname(os.path.realpath(__file__))

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("prefix",help="Prefix to identify current experiment")
    parser.add_argument("dataset_dir",help="A directory containing img (Images) \
                        and cls (GT Segmentation) folder")
    parser.add_argument("--mode",help="base (baseline),adv (adversarial), semi \
                        (semi-supervised)",choices=('base','adv','semi'),default='base')
    parser.add_argument("--lam_adv",help="Weight for Adversarial loss for Segmentation Network training",\
                        default=0.01)
    parser.add_argument("--nogpu",help="Train only on cpus. Helpful for debugging",action='store_true')
    parser.add_argument("--max_epoch",help="Maximum iterations.",default=20,\
                        type=int)
    parser.add_argument("--start_epoch",help="Resume training from this epoch",\
                        default=1,type=int)
    parser.add_argument("--snapshot",help="Snapshot to resume training")
    parser.add_argument("--snapshot_dir",help="Location to store the snapshot", \
                        default=os.path.join(home_dir,'data','snapshots'))
    parser.add_argument("--batch_size",help="Batch size for training",default=10,\
                        type=int)
    parser.add_argument("--val_orig", help="Do Inference on original size image.\
                        Otherwise, crop to 321x321 like in training ",action='store_true')
    args = parser.parse_args()

    # Load the trainloader
    img_transform = [ToTensor(),NormalizeOwn()]
    label_transform = [IgnoreLabelClass(),ToTensorLabel()]
    co_transform = [RandomSizedCrop((321,321))]


    trainset = PascalVOC(home_dir,args.dataset_dir,img_transform=Compose(img_transform), label_transform=Compose(label_transform), \
        co_transform=Compose(co_transform))
    trainloader = DataLoader(trainset,batch_size=args.batch_size,shuffle=True,num_workers=2,drop_last=True)

    print("Training Data Loaded")
    # Load the valoader
    if args.val_orig:
        img_transform = [ZeroPadding(),ToTensor(),NormalizeOwn()]
        label_transform = [IgnoreLabelClass(),ToTensorLabel()]
        co_transform = []
    else:
        img_transform = [ToTensor(),NormalizeOwn()]
        label_transforms = [IgnoreLabelClass(),ToTensorLabel()]
        co_transforms = [RandomSizedCrop((321,321))]

    valset = PascalVOC(home_dir,args.dataset_dir,img_transform=Compose(img_transform), \
        label_transform = Compose(label_transform),co_transform=Compose(co_transform),train_phase=False)

    valoader = DataLoader(valset,batch_size=1)
    print("Validation Data Loaded")
    generator = deeplabv2.Res_Deeplab()
    print("Generator Loaded!")

    optimizer_G = optim.SGD(filter(lambda p: p.requires_grad, \
        generator.parameters()),lr=0.00025,momentum=0.9,\
        weight_decay=0.0001,nesterov=True)
    print("Generator Optimizer Loaded")

    if args.mode == 'adv':
        discriminator = Dis(in_channels=21)
        print("Discriminator Loaded")

        # Assumptions made. Paper doesn't clarify the details
        optimizer_D = optim.SGD(filter(lambda p: p.requires_grad, \
            discriminator.parameters()),lr=0.0001,weight_decay=0.0001,momentum=0.5,nesterov=True)
        print("Discriminator Optimizer Loaded")

    # Load the snapshot if available
    # No pretrained model for the discriminator
    if  args.snapshot and os.path.isfile(args.snapshot):
        print("Snapshot Available at {} ".format(args.snapshot))
        snapshot = torch.load(args.snapshot)
        new_state = generator.state_dict()
        saved_net = {k.partition('module.')[2]: v for i, (k,v) in enumerate(snapshot['state_dict'].items())}
        new_state.update(saved_net)
        generator.load_state_dict(new_state)

    else:
        print("No Snapshot. Loading '{}'".format("MS_DeepLab_resnet_pretrained_COCO_init.pth"))
        saved_net = torch.load(os.path.join(home_dir,'data',\
            'MS_DeepLab_resnet_pretrained_COCO_init.pth'))
        new_state = generator.state_dict()
        saved_net = {k.partition('Scale.')[2]: v for i, (k,v) in enumerate(saved_net.items())}
        new_state.update(saved_net)
        generator.load_state_dict(new_state)

    if not args.nogpu:
        generator = nn.DataParallel(generator).cuda()
        # generator = generator.cuda(0)
        print("Generator Setup for Parallel Training")
        # print("Generator Loaded on device 0")
    else:
        print("No Parallel Training for CPU")


    if args.mode == 'adv':
        if args.nogpu:
            print("No Parallel Training for CPU")
        else:
            discriminator = nn.DataParallel(discriminator).cuda()
            # discriminator = discriminator.cuda(1)
            print("Discriminator Setup for parallel training")
            # print("Discriminator Loaded on device 1")

    best_miou = -1
    for epoch in range(args.start_epoch,args.max_epoch+1):
        generator.train()
        for batch_id, (img,mask,ohmask) in enumerate(trainloader):
            if args.nogpu:
                img,mask,ohmask = Variable(img),Variable(mask,requires_grad=False),\
                                Variable(ohmask,requires_grad=False)
            else:
                img,mask,ohmask = Variable(img.cuda()),Variable(mask.cuda(),requires_grad=False),\
                                Variable(ohmask.cuda(),requires_grad=False)

            # Generate Prediction Map with the Segmentation Network
            out_img_map = generator(img)
            out_img_map = nn.LogSoftmax()(out_img_map)

            #####################
            # Baseline Training #
            ####################
            if args.mode == 'base':
                print("Baseline Training")
                L_seg = nn.NLLLoss2d()(out_img_map,mask)

                i = len(trainloader)*(epoch-1) + batch_id
                poly_lr_scheduler(optimizer_G, 0.00025, i)

                optimizer_G.zero_grad()
                L_seg.backward(retain_variables=True)
                optimizer_G.step()
                print("[{}][{}]Loss: {}".format(epoch,i,L_seg.data[0]))

            #######################
            # Adverarial Training #
            #######################
            if args.mode == 'adv':

                N = out_img_map.size()[0]
                H = out_img_map.size()[2]
                W = out_img_map.size()[3]

                # Generate the Real and Fake Labels
                target_fake = Variable(torch.zeros((N,H,W)).long(),requires_grad=False)
                target_real = Variable(torch.ones((N,H,W)).long(),requires_grad=False)
                if not args.nogpu:
                    target_fake = target_fake.cuda()
                    target_real = target_real.cuda()

                #########################
                # Discriminator Training#
                #########################

                # Train on Real
                conf_map_real = nn.LogSoftmax()(discriminator(ohmask.float()))

                optimizer_D.zero_grad()

                LD_real = nn.NLLLoss2d()(conf_map_real,target_real)
                LD_real.backward()

                # Train on Fake
                conf_map_fake = nn.LogSoftmax()(discriminator(Variable(out_img_map.data)))
                LD_fake = nn.NLLLoss2d()(conf_map_fake,target_fake)
                LD_fake.backward()

                # Update Discriminator weights
                i = len(trainloader)*(epoch-1) + batch_id
                poly_lr_scheduler(optimizer_D, 0.00025, i)

                optimizer_D.step()

                ######################
                # Generator Training #
                #####################
                conf_map_fake = nn.LogSoftmax()(discriminator(out_img_map))
                LG_ce = nn.NLLLoss2d()(out_img_map,mask)
                LG_adv = args.lam_adv * nn.NLLLoss2d()(conf_map_fake,target_real)

                LG_seg = LG_ce.data[0] + LG_adv.data[0]
                optimizer_G.zero_grad()
                LG_ce.backward(retain_variables=True)
                LG_adv.backward()
                poly_lr_scheduler(optimizer_G, 0.00025, i)
                optimizer_G.step()
                print("[{}][{}] LD: {} LG: {}".format(epoch,i,(LD_real + LD_fake).data[0],LG_seg))
        snapshot = {
            'epoch': epoch,
            'state_dict': generator.state_dict(),

        }
        miou = val(generator,valoader)

        snapshot['miou'] = miou
        print("Epoch: {} miou: {}:".format(epoch,miou))
        if miou > best_miou:
            print("Best miou: {}, at epoch: {}".format(miou,epoch))
            best_miou = miou
            torch.save(snapshot,os.path.join(args.snapshot_dir,'{}.pth.tar'.format(args.prefix)))
            print("Snapshot written")


if __name__ == '__main__':
    main()
