from __future__ import unicode_literals
import random
import numpy as np
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
home_dir = os.path.dirname(os.path.realpath(__file__))
def parse_args():

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

    parser.add_argument("--lam_semi",default=0.1,
                        help="Weight for Semi-supervised loss")

    parser.add_argument("--t_semi",default=0.1,type=float,
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

    parser.add_argument("--d_label_smooth",default=0.1,type=float,
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

    parser.add_argument("--wait_semi",default=0,type=int,
                        help="Number of Epochs to wait before using semi-supervised loss")

    parser.add_argument("--split",default=1,type=float)
    args = parser.parse_args()

    return args

'''
    Snapshot the Best Model
'''
def snapshot(model,valoader,epoch,best_miou,snapshot_dir,prefix):
    miou = val(model,valoader)
    snapshot = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'miou': miou
    }
    if miou > best_miou:
        best_miou = miou
        torch.save(snapshot,os.path.join(snapshot_dir,'{}.pth.tar'.format(prefix)))

    print("[{}] Curr mIoU: {:0.4f} Best mIoU: {}".format(epoch,miou,best_miou))
    return best_miou

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
    elif init_net == 'mscoco':

        # TODO: Upload the weights somewhere to use load.url()
        filename = os.path.join(home_dir,'data','MS_DeepLab_resnet_pretrained_COCO_init.pth')
        assert(os.path.isfile(filename))
        saved_net = torch.load(filename)
        new_state = model.state_dict()
        saved_net = {k.partition('Scale.')[2]: v for i, (k,v) in enumerate(saved_net.items())}
        new_state.update(saved_net)
        model.load_state_dict(new_state)

'''
    Baseline Training
'''
def train_base(generator,optimG,trainloader,valoader,args):

    best_miou = -1
    for epoch in range(args.start_epoch,args.max_epoch+1):
        generator.train()
        for batch_id, (img,mask,_) in enumerate(trainloader):

            if args.nogpu:
                img,mask = Variable(img),Variable(mask)
            else:
                img,mask = Variable(img.cuda()),Variable(mask.cuda())

            itr = len(trainloader)*(epoch-1) + batch_id
            cprob = generator(img)
            cprob = nn.LogSoftmax()(cprob)

            Lseg = nn.NLLLoss2d()(cprob,mask)

            optimG = poly_lr_scheduler(optimG, args.g_lr, itr)
            optimG.zero_grad()

            Lseg.backward()
            optimG.step()

            print("[{}][{}]Loss: {:0.4f}".format(epoch,itr,Lseg.data[0]))

        best_miou = snapshot(generator,valoader,epoch,best_miou,args.snapshot_dir,args.prefix)

'''
    Adversarial Training
'''
def train_adv(generator,discriminator,optimG,optimD,trainloader,valoader,args):

    best_miou = -1
    for epoch in range(args.start_epoch,args.max_epoch+1):
        generator.train()
        for batch_id, (img,mask,ohmask) in enumerate(trainloader):
            if args.nogpu:
                img,mask,ohmask = Variable(img),Variable(mask),Variable(ohmask)
            else:
                img,mask,ohmask = Variable(img.cuda()),Variable(mask.cuda()),Variable(ohmask.cuda())
            itr = len(trainloader)*(epoch-1) + batch_id
            cpmap = generator(Variable(img.data,volatile=True))
            cpmap = nn.Softmax2d()(cpmap)

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
            LDr.backward()

            # Train on Fake
            conff = nn.LogSoftmax()(discriminator(Variable(cpmap.data)))
            LDf = nn.NLLLoss2d()(conff,targetf)
            LDf.backward()

            optimD = poly_lr_scheduler(optimD, args.d_lr, itr)
            optimD.step()

            ######################
            # GENERATOR TRAINING #
            #####################
            optimG.zero_grad()
            optimD.zero_grad()
            cmap = generator(img)
            cpmapsmax = nn.Softmax2d()(cmap)
            cpmaplsmax = nn.LogSoftmax()(cmap)
            conff = nn.LogSoftmax()(discriminator(cpmapsmax))


            LGce = nn.NLLLoss2d()(cpmaplsmax,mask)
            LGadv = nn.NLLLoss2d()(conff,targetr)
            LGseg = LGce + args.lam_adv *LGadv

            LGseg.backward()
            poly_lr_scheduler(optimG, args.g_lr, itr)
            optimG.step()

            print("[{}][{}] LD: {:.4f} LDfake: {:.4f} LD_real: {:.4f} LG: {:.4f} LG_ce: {:.4f} LG_adv: {:.4f}"  \
                    .format(epoch,itr,(LDr + LDf).data[0],LDr.data[0],LDf.data[0],LGseg.data[0],LGce.data[0],LGadv.data[0]))
        best_miou = snapshot(generator,valoader,epoch,best_miou,args.snapshot_dir,args.prefix)

'''
    Semi supervised training
'''
def train_semi(generator,discriminator,optimG,optimD,trainloader_l,trainloader_u,valoader,args):
    best_miou = -1
    for epoch in range(args.start_epoch,args.max_epoch+1):
        generator.train()
        trainloader_l_iter = iter(trainloader_l)
        trainloader_u_iter = iter(trainloader_u)
        print("Epoch: {}".format(epoch))
        batch_id = 0
        # Randomly pick labeled or unlabeled data for training
        while(True):
            if random.random() <0.5:
                loader = trainloader_l_iter
                labeled = True
            else:
                loader = trainloader_u_iter
                labeled = False
            # Check if the loader has a batch available
            try:
                img,mask,ohmask = next(loader)
            except:
                # Curr loader doesn't have data
                if labeled:
                    loader = trainloader_u_iter
                    labeled = False
                else:
                    loader = trainloader_l_iter
                    labeled = True

                # Check if the new loader has data
                try:
                    img,mask,ohmask = next(loader)
                except:
                    # Boith loaders exhausted
                    break

            batch_id += 1
            if args.nogpu:
                img,mask,ohmask = Variable(img),Variable(mask),Variable(ohmask)
            else:
                img,mask,ohmask = Variable(img.cuda()),Variable(mask.cuda()),Variable(ohmask.cuda())
            itr = (len(trainloader_u) + len(trainloader_l))*(epoch-1) + batch_id

            if labeled:
                ################################################
                #  Labelled data for Discriminator Training #
                ################################################
                cpmap = generator(Variable(img.data,volatile=True))
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
                confr = nn.LogSoftmax()(discriminator(ohmask.float()))

                optimD.zero_grad()

                if args.d_label_smooth != 0:
                    LDr = (1 - args.d_label_smooth)*nn.NLLLoss2d()(confr,targetr)
                    LDr += args.d_label_smooth * nn.NLLLoss2d()(confr,targetf)
                else:
                    LDr = nn.NLLLoss2d()(confr,targetr)
                LDr.backward()

                # Train on Fake
                conff = nn.LogSoftmax()(discriminator(Variable(cpmap.data)))
                LDf = nn.NLLLoss2d()(conff,targetf)
                LDf.backward()

                LDr_d = LDr.data[0]
                LDf_d = LDf.data[0]
                LD_d = LDr_d + LDf_d
                optimD = poly_lr_scheduler(optimD, args.d_lr, itr)
                optimD.step()

                #####################################
                #  labelled data Generator Training #
                #####################################
                optimG.zero_grad()
                optimD.zero_grad()
                cpmap = generator(img)
                cpmapsmax = nn.Softmax2d()(cpmap)
                cpmaplsmax = nn.LogSoftmax()(cpmap)

                conff = nn.LogSoftmax()(discriminator(cpmapsmax))

                LGce = nn.NLLLoss2d()(cpmaplsmax,mask)
                LGadv = nn.NLLLoss2d()(conff,targetr)

                LGadv_d = LGadv.data[0]
                LGce_d = LGce.data[0]
                LGsemi_d = 0 # No semi-supervised training

                LGadv = args.lam_adv*LGadv

                (LGce + LGadv).backward()
                optimG = poly_lr_scheduler(optimG, args.g_lr, itr)
                optimG.step()

            else:
                #####################################
                # Use unlabelled data to get L_semi #
                #####################################
                # No discriminator training
                LD_d = 0
                LDr_d = 0
                LDf_d = 0
                # Init all loss to 0 for logging ease
                LGsemi_d = 0
                LGce_d = 0
                LGadv_d = 0
                optimG.zero_grad()
                if epoch > args.wait_semi:
                    cpmap = generator(img)
                    cpmapsmax = nn.Softmax2d()(cpmap)

                    conf = discriminator(cpmapsmax)
                    confsmax = nn.Softmax2d()(conf)
                    conflsmax = nn.LogSoftmax()(conf)

                    N = cpmap.size()[0]
                    H = cpmap.size()[2]
                    W = cpmap.size()[3]

                    # Adversarial Loss
                    targetr = Variable(torch.ones((N,H,W)).long())
                    if not args.nogpu:
                        targetr = targetr.cuda()
                    LGadv = nn.NLLLoss2d()(conflsmax,targetr)
                    LGadv_d = LGadv.data[0]
                    # Semi-Supervised Loss

                    hardpred = torch.max(cpmapsmax,1)[1].squeeze(1)

                    idx = np.zeros(cpmap.data.cpu().numpy().shape,dtype=np.uint8)
                    idx = idx.transpose(0, 2, 3, 1)

                    confnp = confsmax[:,1,...].data.cpu().numpy()
                    hardprednp = hardpred.data.cpu().numpy()
                    idx[confnp > args.t_semi] = np.identity(21, dtype=idx.dtype)[hardprednp[ confnp > args.t_semi]]

                    LG = args.lam_adv*LGadv
                    if np.count_nonzero(idx) != 0:
                        cpmaplsmax = nn.LogSoftmax()(cpmap)
                        idx = Variable(torch.from_numpy(idx.transpose(0,3,1,2)).byte().cuda())
                        LGsemi_arr = cpmaplsmax.masked_select(idx)
                        LGsemi = -1*LGsemi_arr.mean()
                        LGsemi_d = LGsemi.data[0]
                        LGsemi = args.lam_semi*LGsemi
                        LG += LGsemi

                    LG.backward()
                    optimG = poly_lr_scheduler(optimG, args.g_lr, itr)
                    optimG.step()
                    # Manually free all variables. Look into details of how variables are freed
                    del idx
                    del confnp
                    del confsmax
                    del hardpred
                    del hardprednp
                    del cpmapsmax
                    del cpmap
                LGseg_d = LGce_d + LGadv_d + LGsemi_d


            # Manually free memory! Later, really understand how computation graphs free variables

            print("[{}][{}] LD: {:.4f} LD_fake: {:.4f} LD_real: {:.4f} LG: {:.4f} LG_ce: {:.4f} LG_adv: {:.4f} LG_semi: {:.4f}"\
                    .format(epoch,itr,LD_d,LDr_d,LDf_d,LGseg_d,LGce_d,LGadv_d,LGsemi_d))

        best_miou = snapshot(generator,valoader,epoch,best_miou,args.snapshot_dir,args.prefix)



def main():

    args = parse_args()

    random.seed(0)
    torch.manual_seed(0)
    if not args.nogpu:
        torch.cuda.manual_seed_all(0)

    if args.no_norm:
        imgtr = [ToTensor()]
    else:
        imgtr = [ToTensor(),NormalizeOwn()]

    labtr = [IgnoreLabelClass(),ToTensorLabel()]
    cotr = [RandomSizedCrop((321,321))]

    trainset_l = PascalVOC(home_dir,args.dataset_dir,img_transform=Compose(imgtr), label_transform=Compose(labtr),co_transform=Compose(cotr),split=args.split,labeled=True)
    trainset_u = PascalVOC(home_dir,args.dataset_dir,img_transform=Compose(imgtr), label_transform=Compose(labtr),co_transform=Compose(cotr),split=args.split,labeled=False)

    trainloader_l = DataLoader(trainset_l,batch_size=args.batch_size,shuffle=True,num_workers=2,drop_last=True)
    trainloader_u = DataLoader(trainset_u,batch_size=args.batch_size,shuffle=True,num_workers=2,drop_last=True)


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
    generator = deeplabv2.ResDeeplab()
    init_weights(generator,args.init_net)

    optimG = optim.SGD(filter(lambda p: p.requires_grad, \
        generator.parameters()),lr=args.g_lr,momentum=0.9,\
        weight_decay=0.0001,nesterov=True)

    if not args.nogpu:
        generator = nn.DataParallel(generator).cuda()

    #################
    # DISCRIMINATOR #
    ################
    if args.mode != "base":
        discriminator = Dis(in_channels=21)
        if args.d_optim == 'adam':
            optimD = optim.Adam(filter(lambda p: p.requires_grad, \
                discriminator.parameters()),lr = args.d_lr,weight_decay=0.0001)
        else:
            optimD = optim.SGD(filter(lambda p: p.requires_grad, \
                discriminator.parameters()),lr=args.d_lr,weight_decay=0.0001,momentum=0.5,nesterov=True)

        if not args.nogpu:
            discriminator = nn.DataParallel(discriminator).cuda()

    if args.mode == 'base':
        train_base(generator,optimG,trainloader_l,valoader,args)
    elif args.mode == 'adv':
        train_adv(generator,discriminator,optimG,optimD,trainloader_l,valoader,args)
    else:
        print("Semi-Supervised training")
        train_semi(generator,discriminator,optimG,optimD,trainloader_l,trainloader_u,valoader,args)

if __name__ == '__main__':

    main()
