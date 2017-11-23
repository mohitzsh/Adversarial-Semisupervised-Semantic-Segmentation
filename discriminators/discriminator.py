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

        self.conv1 = nn.Conv2d(in_channels=self._in_channels,out_channels=64,kernel_size=4,stride=2,padding=1)

        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=1)

        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,stride=2,padding=1)

        self.conv4 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=4,stride=2,padding=1)
        self.relu4 = nn.LeakyReLU(self._negative_slope)

        self.conv5 = nn.Conv2d(in_channels=512,out_channels=1,kernel_size=4,stride=2,padding=1)

    def forward(self,x):
        x= F.LeakyReLU(self.conv1(x),self._negative_slope) # -,-,161,161
        x= F.LeakyReLU(self.conv2(x),self._negative_slope) # -,-,81,81
        x= F.LeakyReLU(self.conv3(x),self._negative_slope) # -,-,41,41
        x= F.LeakyReLU(self.conv4(x),self._negative_slope) # -,-,21,21
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
