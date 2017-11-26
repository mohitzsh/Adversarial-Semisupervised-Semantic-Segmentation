import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class Dis(nn.Module):
    """
        Disriminator Network for the Adversarial Training.
    """
    def __init__(self,in_channels,negative_slope = 0.2):
        super(Dis, self).__init__()
        self._in_channels = in_channels
        self._negative_slope = negative_slope

        self.conv1 = nn.Conv2d(in_channels=self._in_channels,out_channels=64,kernel_size=4,stride=2,padding=2)
        self.relu1 = nn.LeakyReLU(self._negative_slope,inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=2)
        self.relu2 = nn.LeakyReLU(self._negative_slope,inplace=True)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,stride=2,padding=2)
        self.relu3 = nn.LeakyReLU(self._negative_slope,inplace=True)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=4,stride=2,padding=2)
        self.relu4 = nn.LeakyReLU(self._negative_slope,inplace=True)
        self.conv5 = nn.Conv2d(in_channels=512,out_channels=2,kernel_size=4,stride=2,padding=2)

    def forward(self,x):
        x= self.conv1(x) # -,-,161,161
        x = self.relu1(x)
        x= self.conv2(x) # -,-,81,81
        x = self.relu2(x)
        x= self.conv3(x) # -,-,41,41
        x = self.relu3(x)
        x= self.conv4(x) # -,-,21,21
        x = self.relu4(x)
        x = self.conv5(x) # -,-,11,11
        # upsample
        x = F.upsample_bilinear(x,scale_factor=2)
        x = x[:,:,:-1,:-1] # -,-, 21,21

        x = F.upsample_bilinear(x,scale_factor=2)
        x = x[:,:,:-1,:-1] # -,-,41,41

        x = F.upsample_bilinear(x,scale_factor=2)
        x = x[:,:,:-1,:-1] #-,-,81,81

        x = F.upsample_bilinear(x,scale_factor=2)
        x = x[:,:,:-1,:-1] #-,-,161,161

        x = F.upsample_bilinear(x,scale_factor=2)
        x = x[:,:,:-1,:-1] # -,-,321,321

        return x
