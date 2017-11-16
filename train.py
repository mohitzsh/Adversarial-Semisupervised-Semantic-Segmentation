from __future__ import unicode_literals

from collections import OrderedDict
import torch
from datasets.pascalvoc import PascalVOC
import generators.deeplabv2 as deeplabv2
import discriminators.discriminator as dis
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.transforms import RandomSizedCrop, IgnoreLabelClass, ToTensorLabel, NormalizeOwn
from utils.lr_scheduling import poly_lr_scheduler
import torch.nn.functional as F
import torch.nn as nn
from functools import reduce
import torch.optim as optim
import os
import argparse
from torchvision.transforms import ToTensor

def main():

    home_dir = os.path.dirname(os.path.realpath(__file__))

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_dir",help="A directory containing img (Images) \
                        and cls (GT Segmentation) folder")
    parser.add_argument("--max_epoch",help="Maximum iterations.",default=20,\
                        type=int)
    parser.add_argument("--start_epoch",help="Resume training from this epoch",\
                        default=1,type=int)
    parser.add_argument("--snapshot",help="Snapshot to resume training")
    parser.add_argument("--snapshot_dir",help="Location to store the snapshot", \
                        default=os.path.join(home_dir,'data','snapshots'))
    parser.add_argument("--batch_size",help="Batch size for training",default=10,type=int)

    # Add arguments for Optimizer later
    args = parser.parse_args()

    # Define the transforms for the data

    img_transform = transforms.Compose([ToTensor(),NormalizeOwn(dataset='voc')])
    label_transform = transforms.Compose([IgnoreLabelClass(),ToTensorLabel()])

    co_transform = transforms.Compose([RandomSizedCrop((321,321))])
    trainset = PascalVOC(home_dir,args.dataset_dir,img_transform=img_transform, label_transform=label_transform, \
        co_transform=co_transform)
    trainloader = DataLoader(trainset,batch_size=args.batch_size,shuffle=True,num_workers=2)

    print("Dataset setup done!")

    generator = deeplabv2.Res_Deeplab()
    print("Generator setup done!")

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, \
        generator.parameters()),lr=0.00025,momentum=0.9,\
        weight_decay=0.0001,nesterov=True)

    print("Optimizer setup done")
    # Load the snapshot if available
    if  args.snapshot and os.path.isfile(args.snapshot):
        snapshot = torch.load(args.snapshot)
        args.start_epoch = snapshot['epoch']
        saved_net = {k.partition('module.')[2]: v for i, (k,v) in enumerate(snapshot['state_dict'].items())}
        generator.load_state_dict(saved_net)
        optimizer.load_state_dict(snapshot['optimizer'])
        print("Snapshot Available. Resuming Training from iter: {} ".format(args.start_iter))

    else:
        print("No Snapshot. Loading '{}'".format("MS_DeepLab_resnet_pretrained_COCO_init.pth"))
        saved_net = torch.load(os.path.join(home_dir,'data',\
            'MS_DeepLab_resnet_pretrained_COCO_init.pth'))
        new_state = generator.state_dict()
        # Remove Scale. prefix from all the saved_net keys
        saved_net = {k.partition('Scale.')[2]: v for i, (k,v) in enumerate(saved_net.items())}
        new_state.update(saved_net)
        generator.load_state_dict(new_state)

    print('Generator Net created')

    generator = nn.DataParallel(generator).cuda()
    # Setup the optimizer
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, \
        generator.parameters()),lr=0.00025,momentum=0.9,\
        weight_decay=0.0001,nesterov=True)

    logfile = open("log.txt",'w')

    print('Training Going to Start')
    for epoch in range(args.start_epoch,args.max_epoch+1):

        for batch_id, (img,mask) in enumerate(trainloader):
            img,mask = Variable(img.cuda()),Variable(mask.cuda())
            out_img_map = generator(img)
            out_img_map = nn.LogSoftmax()(out_img_map)
            L_ce = nn.NLLLoss2d()
            loss = L_ce(out_img_map,mask.long())
            i = len(trainloader)*(epoch-1) + batch_id
            poly_lr_scheduler(optimizer, 0.00025, i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logfile.write('Epoch {} Batch_id : {} Loss: {:.6f}\n'.format(epoch,batch_id,  loss.data[0]))
            print('Epoch {} Batch_id : {} Loss: {:.6f}\n'.format(epoch,batch_id,  loss.data[0]))
        print("Preparing to Write the snapshot")
        # Flush the log file
        logfile.flush()
        state = {
            'epoch': epoch,
            'state_dict': generator.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        # Write the new snapshot and delete all other snapshots
        curr_snapshot = os.path.join(args.snapshot_dir,'{}.pth.tar'.format(epoch))
        torch.save(state,curr_snapshot)
        filelist = os.listdir(args.snapshot_dir)
        filelist = list(filter(lambda f: f != '{}.pth.tar'.format(epoch), filelist))
        for f in filelist:
            os.remove(os.path.join(args.snapshot_dir,f))
        print("Snapshot written")


if __name__ == '__main__':
    main()
