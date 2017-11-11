from __future__ import unicode_literals

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

MAX_ITR = 1000

def main():
    home_dir = os.path.join(os.environ["HOME"],"adversarial_segmentation")
    pascal_base_dir = os.path.join(os.environ["RCAC_SCRATCH"],"PASCALVOC")
    transform = transforms.Compose([transforms.ToTensor()])
    co_transform = transforms.Compose([RandomSizedCrop((321,321))])
    trainset = PascalVOC(home_dir,pascal_base_dir,transform=transform, co_transform=co_transform)
    print("Trainset created.")
    trainloader = DataLoader(trainset,batch_size=10,shuffle=True,num_workers=2)
    print('TrainLoader created')

    generator = deeplabv2.Res_Deeplab().cuda()

    saved_net = torch.load(os.path.join(home_dir,'data','MS_DeepLab_resnet_pretrained_COCO_init.pth'))
    new_state = generator.state_dict()
    new_state.update(saved_net)
    generator.load_state_dict(new_state)

    print('Generator Net created')

    # Setup the optimizer
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, generator.parameters()),lr=0.00025,momentum=0.9,weight_decay=0.0001,nesterov=True)
    optimizer.zero_grad()

    print('Training Going to Start')
    for iteration in range(0,MAX_ITR):

        for batch_id, (img,mask) in enumerate(trainloader):
            optimizer.zero_grad()
            img,mask = Variable(img.cuda()),Variable(mask.cuda())
            out_img_map = generator(img)
            out_img_map = nn.LogSoftmax()(out_img_map)
            L_ce = nn.NLLLoss2d()
            loss = L_ce(out_img_map,mask.long())
            loss.backward()
            optimizer.step()
        print("Iteration: ", iteration, "Loss: ", loss.data)
        # if iteration % SNAPSHOT_ITER == 0:
            # Take a snapshot of the network

if __name__ == '__main__':
    main()
