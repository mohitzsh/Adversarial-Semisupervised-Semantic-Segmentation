import torch
import torch.nn as nn
import torchvision.models as models

class Discriminator(nn.Module):
    """
        Disriminator Network for the Adversarial Training.
    """
    def __init__(self,target_size,in_channels = 3,negative_slope = 0.2):
        super(Discriminator, self).__init__()
        self._in_channels = in_channels
        self._negative_slope = negative_slope
        self._target_size = target_size

        self.conv1 = nn.conv2D(self._in_channels,out_channels=64,kernel_size=4,stride=2)
        self.relu1 = nn.LeakyReLU(self._negative_slope)

        self.conv2 = nn.conv2D(in_channels=64,out_channels=128,kernel_size=4,stride=2)
        self.relu2 = nn.LeakyReLU(self._negative_slope)

        self.conv3 = nn.conv2D(in_channels=128,out_channels=256,kernel_size=4,stride=2)
        self.relu3 = nn.LeakyReLU(self._negative_slope)

        self.conv4 = nn.conv2D(in_channels=256,out_channels=512,kernel_size=4,stride=2)
        self.relu4 = nn.LeakyReLU(self._negative_slope)

        # The output layer (at each pixel) represent if that pixel was sampled from ground truth(p=1) or from the Generator(p=0)
        self.conv5 = nn.conv2D(in_channels=512,out_channels=1,kernel_size=4,stride=2)


    def forward(self,x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.conv5(x)

        # Sample it to some size

        return x
