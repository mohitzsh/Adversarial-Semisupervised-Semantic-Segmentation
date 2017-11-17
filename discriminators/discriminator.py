import torch
import torch.nn as nn
import torchvision.models as models

class Discrim(nn.Module):
    """
        Disriminator Network for the Adversarial Training.
    """
    def __init__(self,in_channels = 3,negative_slope = 0.2):
        super(Discriminator, self).__init__()
        self._in_channels = in_channels
        self._negative_slope = negative_slope

        self.conv1 = nn.Conv2D(in_channels=elf._in_channels,out_channels=64,kernel_size=4,stride=2)
        self.relu1 = nn.LeakyReLU(self._negative_slope)

        self.conv2 = nn.Conv2D(in_channels=64,out_channels=128,kernel_size=4,stride=2)
        self.relu2 = nn.LeakyReLU(self._negative_slope)

        self.conv3 = nn.Conv2D(in_channels=128,out_channels=256,kernel_size=4,stride=2)
        self.relu3 = nn.LeakyReLU(self._negative_slope)

        self.conv4 = nn.Conv2D(in_channels=256,out_channels=512,kernel_size=4,stride=2)
        self.relu4 = nn.LeakyReLU(self._negative_slope)

        self.conv5 = nn.Conv2D(in_channels=512,out_channels=1,kernel_size=4,stride=2)

        #upsample to the original image size
        self.up_conv5 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=4,stride=2)
        self.up_conv4 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=4,stride=2)
        self.up_conv3 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=4,stride=2)
        self.up_conv2 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=4,stride=2)
        self.up_conv1 = nn.ConvTranspose2d(in_channels=1,out_channels=1,kernel_size=4,stride=2)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.up_conv5(x)
        x = self.up_conv4(x)
        x = self.up_conv3(x)
        x = self.up_conv2(x)
        x = self.up_conv1(x)

        return x
