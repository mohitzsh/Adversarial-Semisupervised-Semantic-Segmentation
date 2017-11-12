from __future__ import unicode_literals

from collections import OrderedDict
import torch
from datasets.pascalvoc import PascalVOC
import generators.deeplabv2 as deeplabv2
import discriminators.discriminator as dis
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.transforms import RandomSizedCrop
import torch.nn.functional as F
import torch.nn as nn
from functools import reduce
import torch.optim as optim
import os
import argparse

def main():

    home_dir = os.path.dirname(os.path.realpath(__file__))

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_dir",help="A directory containing img (Images) \
                        and cls (GT Segmentation) folder")
    parser.add_argument("--max_iter",help="Maximum iterations.",default=20000,\
                        type=int)
    parser.add_argument("--start_iter",help="Resume training from this iteration",\
                        default=1,type=int)
    parser.add_argument("--snapshot",help="Snapshot to resume training")
    parser.add_argument("--snapshot_iter",help="Iterations for taking snapshot",default=5000,type=int)
    parser.add_argument("--snapshot_dir",help="Location to store the snapshot", \
                        default=os.path.join(home_dir,'data','snapshots'))
    #Try to see if available gpus can be detected at runtime
    parser.add_argument("--gpu",help="GPU to use for training",default=0,type=int)

    # Add arguments for Optimizer later
    args = parser.parse_args()

    # Define the transforms for the data
    transform = transforms.Compose([transforms.ToTensor()])
    co_transform = transforms.Compose([RandomSizedCrop((321,321))])

    trainset = PascalVOC(home_dir,args.dataset_dir,transform=transform, \
        co_transform=co_transform)
    trainloader = DataLoader(trainset,batch_size=10,shuffle=True,num_workers=2)

    print("Dataset setup done!")
    # Prepare the Generator
    generator = deeplabv2.Res_Deeplab().cuda(args.gpu)
    print("Generator setup done!")
    # Prepare the optimizer
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, \
        generator.parameters()),lr=0.00025,momentum=0.9,\
        weight_decay=0.0001,nesterov=True)

    print("Optimizer setup done")
    # Load the snapshot if available
    if  args.snapshot and os.path.isfile(args.snapshot):
        snapshot = torch.load(args.snapshot)
        args.start_iter = snapshot['iter']
        generator.load_state_dict(snapshot['state_dict'])
        optimizer.load_state_dict(snapshot['optimizer'])
        print("Snapshot Available. Resuming Training from iter: {} ".format(args.start_iter))

    else:
        print("No Snapshot. Loading {}'".format("MS_DeepLab_resnet_pretrained_COCO_init.pth"))
        saved_net = torch.load(os.path.join(home_dir,'data',\
            'MS_DeepLab_resnet_pretrained_COCO_init.pth'))
        new_state = generator.state_dict()
        # Remove Scale. prefix from all the saved_net keys
        saved_net = {k.partition('Scale.')[2]: v for i, (k,v) in enumerate(saved_net.items())}
        new_state.update(saved_net)
        generator.load_state_dict(new_state)

    print('Generator Net created')

    # Setup the optimizer
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, \
        generator.parameters()),lr=0.00025,momentum=0.9,\
        weight_decay=0.0001,nesterov=True)



    print('Training Going to Start')
    for iteration in range(args.start_iter,args.max_iter+1):

        for batch_id, (img,mask) in enumerate(trainloader):
            img,mask = Variable(img.cuda(args.gpu)),Variable(mask.cuda(args.gpu))
            out_img_map = generator(img)
            out_img_map = nn.LogSoftmax()(out_img_map)
            L_ce = nn.NLLLoss2d()
            loss = L_ce(out_img_map,mask.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Iter: {} Loss: {:.6f}'.format(iteration,  loss.data[0]))
        if iteration % args.snapshot_iter == 0:
            state = {
                'iter': iteration,
                'state_dict': generator.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(state,os.path.join(args.snapshot_dir,'{}.pth.tar'.format(iteration)))

if __name__ == '__main__':
    main()
