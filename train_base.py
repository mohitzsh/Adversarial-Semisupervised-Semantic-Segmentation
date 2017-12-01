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
import torch.utils.model_zoo as model_zoo

def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    model = model.module
    b = []

    b.append(model.conv1)
    b.append(model.bn1)
    b.append(model.layer1)
    b.append(model.layer2)
    b.append(model.layer3)
    b.append(model.layer4)


    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """

    # Get the model from DataParallel
    model = model.module
    b = []
    b.append(model.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i

def lr_poly(base_lr, iter,max_iter,power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def main():

    home_dir = os.path.dirname(os.path.realpath(__file__))

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("prefix",
                        help="Prefix to identify current experiment")

    parser.add_argument("dataset_dir",
                        help="A directory containing img (Images) and cls (GT Segmentation) folder")

    parser.add_argument("--mode", choices=('base','adv','semi'),default='base',
                        help="base (baseline),adv (adversarial), semi (semi-supervised)")

    parser.add_argument("--lam_adv",default=0.01,
                        help="Weight for Adversarial loss for Segmentation Network training")

    parser.add_argument("--nogpu",action='store_true',
                        help="Train only on cpus. Helpful for debugging")

    parser.add_argument("--max_epoch",default=20,type=int,
                        help="Maximum iterations.")

    parser.add_argument("--start_epoch",default=1,type=int,
                        help="Resume training from this epoch")

    parser.add_argument("--snapshot",
                        help="Snapshot to resume training")

    parser.add_argument("--snapshot_dir",default=os.path.join(home_dir,'data','snapshots'),
                        help="Location to store the snapshot")

    parser.add_argument("--batch_size",default=10,type=int,
                        help="Batch size for training")

    parser.add_argument("--val_orig",action='store_true',
                        help="Do Inference on original size image. Otherwise, crop to 321x321 like in training ")

    parser.add_argument("--d_label_smooth",default=0.25,type=float,
                        help="Label smoothing for real images in Discriminator")

    parser.add_argument("--d_optim",choices=('sgd','adam'),default='sgd',
                        help="Discriminator Optimizer.")

    parser.add_argument("--no_norm",action='store_true',
                        help="No Normalizaion on the Images")

    parser.add_argument("--init_net",choices=('imagenet','mscoco'),default='mscoco',
                        help="Pretrained Net for Segmentation Network")

    args = parser.parse_args()

    # Load the trainloader
    if args.no_norm:
        img_transform = [ToTensor()]
    else:
        img_transform = [ToTensor(),NormalizeOwn()]

    label_transform = [IgnoreLabelClass(),ToTensorLabel()]
    co_transform = [RandomSizedCrop((321,321))]


    trainset = PascalVOC(home_dir,args.dataset_dir,img_transform=Compose(img_transform), label_transform=Compose(label_transform), \
        co_transform=Compose(co_transform))
    trainloader = DataLoader(trainset,batch_size=args.batch_size,shuffle=True,num_workers=2,drop_last=True)

    # Load the valoader
    if args.val_orig:
        if args.no_norm:
            img_transform = [ZeroPadding(),ToTensor()]
        else:
            img_transform = [ZeroPadding(),ToTensor(),NormalizeOwn()]
        label_transform = [IgnoreLabelClass(),ToTensorLabel()]
        co_transform = []
    else:
        if args.no_norm:
            img_transform = [ToTensor()]
        else:
            img_transform = [ToTensor(),NormalizeOwn()]
        label_transforms = [IgnoreLabelClass(),ToTensorLabel()]
        co_transforms = [RandomSizedCrop((321,321))]

    valset = PascalVOC(home_dir,args.dataset_dir,img_transform=Compose(img_transform), \
        label_transform = Compose(label_transform),co_transform=Compose(co_transform),train_phase=False)

    valoader = DataLoader(valset,batch_size=1)
    generator = deeplabv2.Res_Deeplab()

    if args.init_net == 'imagenet':

        # Pretrain on ImageNet
        inet_weights = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        del inet_weights['fc.weight']
        del inet_weights['fc.bias']
        state = generator.state_dict()
        state.update(inet_weights)
        generator.load_state_dict(state)
    elif args.init_net == 'mscoco':

        # Pretrain on MSCOCO
        filename = os.path.join(home_dir,'data','MS_DeepLab_resnet_pretrained_COCO_init.pth')
        assert(os.path.isfile(filename))
        saved_net = torch.load(filename)
        new_state = generator.state_dict()
        saved_net = {k.partition('Scale.')[2]: v for i, (k,v) in enumerate(saved_net.items())}
        new_state.update(saved_net)
        generator.load_state_dict(new_state)

    optimizer_G = optim.SGD(filter(lambda p: p.requires_grad, \
        generator.parameters()),lr=0.00025,momentum=0.9,\
        weight_decay=0.0001,nesterov=True)

    if args.mode == 'adv':
        discriminator = Dis(in_channels=21)
        print("Discriminator Loaded")

        # Assumptions made. Paper doesn't clarify the details
        if args.d_optim == 'adam':
            optimizer_D = optim.Adam(filter(lambda p: p.requires_grad, \
                discriminator.parameters()),lr = 0.0001)
        else:
            optimizer_D = optim.SGD(filter(lambda p: p.requires_grad, \
                discriminator.parameters()),lr=0.0001,weight_decay=0.0001,momentum=0.5,nesterov=True)

        if not args.nogpu:
            discriminator = nn.DataParallel(discriminator).cuda()

    if not args.nogpu:
        generator = nn.DataParallel(generator).cuda()


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


            #####################
            # Baseline Training #
            ####################
            if args.mode == 'base':
                out_img_map = generator(img)
                out_img_map = nn.LogSoftmax()(out_img_map)

                L_seg = nn.NLLLoss2d()(out_img_map,mask)

                i = len(trainloader)*(epoch-1) + batch_id
                poly_lr_scheduler(optimizer_G, 0.00025, i)
                lr_ = lr_poly(0.00025,i,20000,0.9)

                # Might Need to change this
                optimizer_G = optim.SGD([{'params': get_1x_lr_params(generator), 'lr': lr_ },\
                                        {'params': get_10x_lr_params(generator), 'lr': 10*lr_} ], \
                                        lr = lr_, momentum = 0.9,weight_decay = 0.0001,nesterov=True)

                optimizer_G.zero_grad()
                L_seg.backward()
                optimizer_G.step()
                print("[{}][{}]Loss: {}".format(epoch,i,L_seg.data[0]))

            #######################
            # Adverarial Training #
            #######################
            if args.mode == 'adv':

                out_img_map = generator(Variable(img.data,volatile=True))
                out_img_map = nn.Softmax2d()(out_img_map)

                # print("First Forward pass on generator")

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

                # Perform Label smoothing
                if args.d_label_smooth != 0:
                    LD_real = (1 - args.d_label_smooth)*nn.NLLLoss2d()(conf_map_real,target_real)
                    LD_real += args.d_label_smooth * nn.NLLLoss2d()(conf_map_real,target_fake)
                else:
                    LD_real = nn.NLLLoss2d()(conf_map_real,target_real)
                LD_real.backward()

                # Train on Fake
                conf_map_fake = nn.LogSoftmax()(discriminator(Variable(out_img_map.data)))
                # print("Second forward pass on discriminator")
                LD_fake = nn.NLLLoss2d()(conf_map_fake,target_fake)
                LD_fake.backward()


                # Update Discriminator weights
                i = len(trainloader)*(epoch-1) + batch_id
                poly_lr_scheduler(optimizer_D, 0.00025, i)

                optimizer_D.step()

                ######################
                # Generator Training #
                #####################

                out_img_map = generator(img)
                out_img_map_smax = nn.Softmax2d()(out_img_map)
                out_img_map_lsmax = nn.LogSoftmax()(out_img_map)

                conf_map_fake = nn.LogSoftmax()(discriminator(out_img_map_smax))


                LG_ce = nn.NLLLoss2d()(out_img_map_lsmax,mask)
                LG_adv = nn.NLLLoss2d()(conf_map_fake,target_real)

                LG_seg = LG_ce + args.lam_adv *LG_adv
                optimizer_G.zero_grad()
                LG_seg.backward()

                poly_lr_scheduler(optimizer_G, 0.00025, i)
                optimizer_G.step()
                print("[{}][{}] LD: {:.4f} LD_fake: {:.4f} LD_real: {:.4f} LG: {:.4f} LG_ce: {:.4f} LG_adv: {:.4f}"\
                        .format(epoch,i,(LD_real + LD_fake).data[0],LD_real.data[0],LD_fake.data[0],LG_seg.data[0],LG_ce.data[0],LG_adv.data[0]))
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
