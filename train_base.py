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
import random
import numpy as np

def parse_args():
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

    parser.add_argument("--lam_semi",default=0.2,
                        help="Weight for Semi-supervised loss")

    parser.add_argument("--t_semi",default=0.2,
                        help="Threshold for self-taught learning")

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

    parser.add_argument("--d_lr",default=0.0001,type=float,
                        help="lr for discriminator")

    parser.add_argument("--g_lr",default=0.00025,type=float,
                        help="lr for generator")

    parser.add_argument("--seed",default=1,type=int,
                        help="Seed for random numbers used in semi-supervised training")

    args = parser.parse_args()

    return args

'''
    Snapshot the Best Model
'''
def snapshot(model,valoader,epoch,best_miou):
    miou = val(model,valoader)
    snapshot = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'miou': miou
    }
    if miou > best_miou:
        best_miou = miou
        torch.save(snapshot,os.path.join(args.snapshot_dir,'{}.pth.tar'.format(args.prefix)))

    print("[{}] Curr mIoU: {:0.4f} Best mIoU: {}".format(epoch,miou,best_miou))

    return miou

'''
    Use PreTrained Model for Initial Weights
'''
def init_weights(model,init_net):
    if init_net == 'imagenet':
        # Pretrain on ImageNet
        inet_weights = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        del inet_weights['fc.weight']
        del inet_weights['fc.bias']
        state = model.state_dict()
        state.update(inet_weights)
        model.load_state_dict(state)
    elif args.init_net == 'mscoco':

        # TODO: Upload the weights somewhere to use load.url()
        filename = os.path.join(home_dir,'data','MS_DeepLab_resnet_pretrained_COCO_init.pth')
        assert(os.path.isfile(filename))
        saved_net = torch.load(filename)
        new_state = model.state_dict()
        saved_net = {k.partition('Scale.')[2]: v for i, (k,v) in enumerate(saved_net.items())}
        new_state.update(saved_net)
        model.load_state_dict(new_state)

    return model

'''
    Baseline Training
'''
def train_base(args):

    #######################
    # Training Dataloader #
    #######################

    if args.no_norm:
        imgtr = [ToTensor()]
    else:
        imgtr = [ToTensor(),NormalizeOwn()]

    labtr = [IgnoreLabelClass(),ToTensorLabel()]
    cotr = [RandomSizedCrop((321,321))]

    trainset = PascalVOC(home_dir,args.dataset_dir,img_transform=Compose(imgtr), label_transform=Compose(labtr), \
        co_transform=Compose(cotr))
    trainloader = DataLoader(trainset,batch_size=args.batch_size,shuffle=True,num_workers=2,drop_last=True)

    #########################
    # Validation Dataloader #
    ########################
    if args.val_orig:
        if args.no_norm:
            imgtr = [ZeroPadding(),ToTensor()]
        else:
            imgtr = [ZeroPadding(),ToTensor(),NormalizeOwn()]
        labtr = [IgnoreLabelClass(),ToTensorLabel()]
        cotr = []
    else:
        if args.no_norm:
            imgtr = [ToTensor()]
        else:
            imgtr = [ToTensor(),NormalizeOwn()]
        labtr = [IgnoreLabelClass(),ToTensorLabel()]
        cotr = [RandomSizedCrop((321,321))]

    valset = PascalVOC(home_dir,args.dataset_dir,img_transform=Compose(imgtr), \
        label_transform = Compose(labtr),co_transform=Compose(cotr),train_phase=False)
    valoader = DataLoader(valset,batch_size=1)

    model = deeplabv2.ResDeeplab()
    init_weights(model)

    optimG = optim.SGD(filter(lambda p: p.requires_grad, \
        model.parameters()),lr=args.g_lr,momentum=0.9,\
        weight_decay=0.0001,nesterov=True)

    if not args.nogpu:
        model = nn.DataParallel(model).cuda()

    best_miou = -1
    for epoch in range(args.start_epoch,args.max_epoch+1):
        model.train()
        for batch_id, (img,mask,_) in enumerate(trainloader):

            if args.nogpu:
                img,mask = Variable(img),Variable(mask)
            else:
                img,mask = Variable(img.cuda()),Variable(mask.cuda())

            itr = len(trainloader)*(epoch-1) + batch_id
            cprob = generator(img)
            cprob = nn.LogSoftmax()(cprob)

            Lseg = nn.NLLLoss2d()(cprob,mask)

            poly_lr_scheduler(optimG, args.g_lr, itr)
            optimG.zero_grad()

            Lseg.backward()
            optimG.step()

            print("[{}][{}]Loss: {:0.4f}".format(epoch,i,Lseg.data[0]))

        snapshot(model,valoader,epoch,best_miou)

'''
    Adversarial Training
'''
def train_adv(args):

'''
    Semi-Supervised Training
'''
    if args.no_norm:
        imgtr = [ToTensor()]
    else:
        imgtr = [ToTensor(),NormalizeOwn()]

    labtr = [IgnoreLabelClass(),ToTensorLabel()]
    cotr = [RandomSizedCrop((321,321))]

    trainset = PascalVOC(home_dir,args.dataset_dir,img_transform=Compose(imgtr), label_transform=Compose(labtr), \
        co_transform=Compose(cotr))
    trainloader = DataLoader(trainset,batch_size=args.batch_size,shuffle=True,num_workers=2,drop_last=True)

    #########################
    # Validation Dataloader #
    ########################
    if args.val_orig:
        if args.no_norm:
            imgtr = [ZeroPadding(),ToTensor()]
        else:
            imgtr = [ZeroPadding(),ToTensor(),NormalizeOwn()]
        labtr = [IgnoreLabelClass(),ToTensorLabel()]
        cotr = []
    else:
        if args.no_norm:
            imgtr = [ToTensor()]
        else:
            imgtr = [ToTensor(),NormalizeOwn()]
        labtr = [IgnoreLabelClass(),ToTensorLabel()]
        cotr = [RandomSizedCrop((321,321))]

    valset = PascalVOC(home_dir,args.dataset_dir,img_transform=Compose(imgtr), \
        label_transform = Compose(labtr),co_transform=Compose(cotr),train_phase=False)
    valoader = DataLoader(valset,batch_size=1)

    #############
    # GENERATOR #
    #############
    gen = deeplabv2.ResDeeplab()
    optimG = optim.SGD(filter(lambda p: p.requires_grad, \
        gen.parameters()),lr=args.g_lr,momentum=0.9,\
        weight_decay=0.0001,nesterov=True)

    if not args.nogpu:
        gen = nn.DataParallel(gen).cuda()

    #################
    # DISCRIMINATOR #
    ################
    dis = Dis(in_channels=21)
    if args.d_optim == 'adam':
        optimD = optim.Adam(filter(lambda p: p.requires_grad, \
            dis.parameters()),lr = args.d_lr)
    else:
        optimD = optim.SGD(filter(lambda p: p.requires_grad, \
            dis.parameters()),lr=args.d_lr,weight_decay=0.0001,momentum=0.5,nesterov=True)

    if not args.nogpu:
        dis = nn.DataParallel(dis).cuda()

    #############
    # TRAINING  #
    #############
    best_miou = -1
    for epoch in range(args.start_epoch,args.max_epoch+1):
        generator.train()
        for batch_id, (img,mask,ohmask) in enumerate(trainloader):
            if args.nogpu:
                img,mask,ohmask = Variable(img),Variable(mask),Variable(ohmask)
            else:
                img,mask,ohmask = Variable(img.cuda()),Variable(mask.cuda()),Variable(ohmask.cuda())

            cpmap = generator(Variable(img.data,volatile=True))
            cpmap = nn.Softmax2d()(out_img_map)

            N = cpmap.size()[0]
            H = cpmap.size()[2]
            W = cpmap.size()[3]

            # Generate the Real and Fake Labels
            targetf = Variable(torch.zeros((N,H,W)).long(),requires_grad=False)
            targetr = Variable(torch.ones((N,H,W)).long(),requires_grad=False)
            if not args.nogpu:
                targetf = targetf.cuda()
                targetr = targetr.cuda()

            ##########################
            # DISCRIMINATOR TRAINING #
            ##########################
            optimD.zero_grad()

            # Train on Real
            confr = nn.LogSoftmax()(discriminator(ohmask.float()))
            if args.d_label_smooth != 0:
                LDr = (1 - args.d_label_smooth)*nn.NLLLoss2d()(confr,targetr)
                LDr += args.d_label_smooth * nn.NLLLoss2d()(confr,targetf)
            else:
                LDr = nn.NLLLoss2d()(confr,targetr)
            LDreal.backward()

            # Train on Fake
            conff = nn.LogSoftmax()(dis(Variable(cpmap.data)))
            LDf = nn.NLLLoss2d()(conff,targetf)
            LDf.backward()

            poly_lr_scheduler(optimD, args.d_lr, i)
            optimD.step()

            ######################
            # GENERATOR TRAINING #
            #####################
            optimG.zero_grad()

            cmap = gen(img)
            cpmapsmax = nn.Softmax2d()(cmap)
            cpmaplsmax = nn.LogSoftmax()(cmap)
            conff = nn.LogSoftmax()(dis(cpmapsmax))


            LGce = nn.NLLLoss2d()(cpmaplsmax,mask)
            LGadv = nn.NLLLoss2d()(conff,targetr)
            LGseg = LGce + args.lam_adv *LGadv

            LGseg.backward()
            poly_lr_scheduler(optimG, args.g_lr, i)
            optimG.step()

            print("[{}][{}] LD: {:.4f} LDfake: {:.4f} LD_real: {:.4f} LG: {:.4f} LG_ce: {:.4f} LG_adv: {:.4f}"  \
                    .format(epoch,i,(LDr + LDf).data[0],LDr.data[0],LDf.data[0],LGseg.data[0],LGce.data[0],LGadv.data[0]))
        snapshot(gen,valoader,epoch,best_miou)

def train_semi(args):
    # TODO: Make it more generic to include for other splits
    args.batch_size = args.batch_size*2

    if args.no_norm:
        imgtr = [ToTensor()]
    else:
        imgtr = [ToTensor(),NormalizeOwn()]

    labtr = [IgnoreLabelClass(),ToTensorLabel()]
    cotr = [RandomSizedCrop((321,321))]

    trainset = PascalVOC(home_dir,args.dataset_dir,img_transform=Compose(imgtr), label_transform=Compose(labtr), \
        co_transform=Compose(cotr))
    trainloader = DataLoader(trainset,batch_size=args.batch_size,shuffle=True,num_workers=2,drop_last=True)

    #########################
    # Validation Dataloader #
    ########################
    if args.val_orig:
        if args.no_norm:
            imgtr = [ZeroPadding(),ToTensor()]
        else:
            imgtr = [ZeroPadding(),ToTensor(),NormalizeOwn()]
        labtr = [IgnoreLabelClass(),ToTensorLabel()]
        cotr = []
    else:
        if args.no_norm:
            imgtr = [ToTensor()]
        else:
            imgtr = [ToTensor(),NormalizeOwn()]
        labtr = [IgnoreLabelClass(),ToTensorLabel()]
        cotr = [RandomSizedCrop((321,321))]

    valset = PascalVOC(home_dir,args.dataset_dir,img_transform=Compose(imgtr), \
        label_transform = Compose(labtr),co_transform=Compose(cotr),train_phase=False)
    valoader = DataLoader(valset,batch_size=1)
    #############
    # GENERATOR #
    #############
    gen = deeplabv2.ResDeeplab()
    optimG = optim.SGD(filter(lambda p: p.requires_grad, \
        gen.parameters()),lr=args.g_lr,momentum=0.9,\
        weight_decay=0.0001,nesterov=True)

    if not args.nogpu:
        gen = nn.DataParallel(gen).cuda()

    #################
    # DISCRIMINATOR #
    ################
    dis = Dis(in_channels=21)
    if args.d_optim == 'adam':
        optimD = optim.Adam(filter(lambda p: p.requires_grad, \
            dis.parameters()),lr = args.d_lr)
    else:
        optimD = optim.SGD(filter(lambda p: p.requires_grad, \
            dis.parameters()),lr=args.d_lr,weight_decay=0.0001,momentum=0.5,nesterov=True)

    if not args.nogpu:
        dis = nn.DataParallel(dis).cuda()

    ############
    # TRAINING #
    ############
    for epoch in range(args.start_epoch,args.max_epoch+1):
        generator.train()
        for batch_id, (img,mask,ohmask) in enumerate(trainloader):
            if args.nogpu:
                img,mask,ohmask = Variable(img),Variable(mask),Variable(ohmask)
            else:
                img,mask,ohmask = Variable(img.cuda()),Variable(mask.cuda()),Variable(ohmask.cuda())

        ## TODO: Extend random interleaving for split of any size
        mid  = args.batch_size // 2
        img1,mask1,ohmask1 = img[0:mid,...],mask[0:mid,...],ohmask[0:mid,...]
        img2,mask2,ohmask2 = img[mid:,...],mask[mid:,...],ohmask[mid:,...]

        # Random Interleaving
        if random.random() <0.5:
            imgl,maskl,ohmaskl = img1,mask1,ohmask1
            imgu,masku,ohmasku = img2,mask2,ohmask2
        else:
            imgu,masku,ohmasku = img1,mask1,ohmask1
            imgl,maskl,ohmaskl = img2,mask2,ohmask2

        ################################################
        #  Labelled data for Discriminator Training #
        ################################################
        cpmap = gen(Variable(imgl.data,volatile=True))
        cpmap = nn.Softmax2d()(cpmap)

        N = cpmap.size()[0]
        H = cpmap.size()[2]
        W = cpmap.size()[3]

        # Generate the Real and Fake Labels
        targetf = Variable(torch.zeros((N,H,W)).long())
        targetr = Variable(torch.ones((N,H,W)).long())
        if not args.nogpu:
            targetf = targetf.cuda()
            targetr = targetr.cuda()

        # Train on Real
        confr = nn.LogSoftmax()(discriminator(ohmaskl.float()))
        optimD.zero_grad()
        if args.d_label_smooth != 0:
            LDr = (1 - args.d_label_smooth)*nn.NLLLoss2d()(confr,targetr)
            LDr += args.d_label_smooth * nn.NLLLoss2d()(confr,targetf)
        else:
            LDr = nn.NLLLoss2d()(confr,targetr)
        LDr.backward()

        # Train on Fake
        conff = nn.LogSoftmax()(dis(Variable(cpmap.data)))
        LDf = nn.NLLLoss2d()(conff,targetf)
        LDf.backward()

        poly_lr_scheduler(optimD, args.d_lr, i)
        optimD.step()

        ###########################################
        #  labelled data Generator Training       #
        ###########################################
        optimG.zero_grad()

        cpmap = generator(imgl)
        cpmapsmax = nn.Softmax2d()(out_img_map)
        cpmaplsmax = nn.LogSoftmax()(out_img_map)

        conff = nn.LogSoftmax()(dis(cpmapsmax))

        LGce = nn.NLLLoss2d()(cpmaplsmax,maskl)
        LGadv = nn.NLLLoss2d()(conff,targetr)

        LGadv_d = LGadv.data[0]
        LGce_d = LGce.data[0]

        LGadv = args.lam_adv*LG_adv

        (LGce + LGadv).backward()
        #####################################
        # Use unlabelled data to get L_semi #
        #####################################

        cpmap = generator(imgu)
        softpred = nn.Softmax2d()(cpmap)
        hardpred = torch.max(softpred,1)[1].squeeze(1)
        conf = nn.Softmax2d()(dis(Variable(softpred.data,volatile=True)))

        idx = np.zeros(cpmap.data.cpu().numpy().shape,dtype=np.uint8)
        idx = idx.transpose(0, 2, 3, 1)

        confnp = conf_map[:,1,...].data.cpu().numpy()
        hardprednp = hardpred.data.cpu().numpy()
        idx[conf_mapn > args.t_semi] = np.identity(21, dtype=idx.dtype)[hardprednp[ conf_mapn > args.t_semi]]

        if np.count_nonzero(idx) != 0:
            cpmaplsmax = nn.LogSoftmax()(cpmap)
            idx = Variable(torch.from_numpy(idx).byte().cuda())
            LGsemi_arr = cpmaplsmax.masked_select(idx)
            LGsemi = -1*LGsemi_arr.mean()
            LGsemi_d = LGsemi.data[0]
            LGsemi = args.lam_semi*LGsemi
            LGsemi.backward()
        else:
            LGsemi_d = 0
        LGseg_d = LGce_d + LGadv_d + LGsemi_d

        poly_lr_scheduler(optimizer_G, args.g_lr, i)
        optimG.step()

        # Manually free memory! Later, really understand how computation graphs work
        del idx
        del conf
        del confnp
        del hardpred
        del softpred
        del hardpredn
        del cpmap
        print("[{}][{}] LD: {:.4f} LD_fake: {:.4f} LD_real: {:.4f} LG: {:.4f} LG_ce: {:.4f} LG_adv: {:.4f} LG_semi: {:.4f}"\
                .format(epoch,i,(LDr + LDf).data[0],LDr.data[0],LDf.data[0],LGseg_d,LGce_d,LGadv_d,LGsemi_d))

    snapshot(gen,valoader,epoch,best_miou)



def main():

    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.nogpu:
        torch.cuda.manual_seed_all(args.seed)

    if args.mode == 'base':
        train_base(args)
    elif args.mode == 'adv':
        train_adv(args)
    else
        train_semi(args)


if __name__ == '__main__':

    main()
