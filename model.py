from __future__ import print_function

import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torchsummary import summary

class ConvLayer(nn.Module):
    """Basic 'conv' layer, including:
     A Conv2D layer with desired channels and kernel size,
     A batch-norm layer,
     and A leakyReLu layer with neg_slope of 0.1.
     (Didn't find too much resource what neg_slope really is.
     By looking at the darknet source code, it is confirmed the neg_slope=0.1.
     Ref: https://github.com/pjreddie/darknet/blob/master/src/activations.h)
     Please note here we distinguish between Conv2D layer and Conv layer."""

    def __init__(self, in_channels, 
                    out_channels, 
                    kernel_size, 
                    stride=1, 
                    lrelu_neg_slope=0.1):
        super(ConvLayer, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=lrelu_neg_slope)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.lrelu(out)

        return out

class FastYOLO(nn.Module):
    def __init__(self, last_layer=30):
        super(FastYOLO, self).__init__()

        self.conv0 = ConvLayer(3, 16, 3, 1)
        self.conv2 = ConvLayer(16, 32, 3, 1)
        self.conv4 = ConvLayer(32, 64, 3, 1)
        self.conv6 = ConvLayer(64, 128, 3, 1)
        self.conv8 = ConvLayer(128, 256, 3, 1)
        self.conv10 = ConvLayer(256, 512, 3, 1)
        self.conv12 = ConvLayer(512, 1024, 3, 1)
        self.conv13 = ConvLayer(1024, 1024, 3, 1)
        self.conv14 = ConvLayer(1024, last_layer, 1, 1)

    def forward(self,x):
        x = self.conv0(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv6(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv8(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv10(x)
        x = F.max_pool2d(x, 2, 1)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)

        return x

class CharacterSeg(nn.Module):
    def __init__(self):
        super(CharacterSeg, self).__init__()

        self.conv1 = ConvLayer(3, 32, 3, 1)
        self.conv3 = ConvLayer(32, 64, 3, 1)
        self.conv5 = ConvLayer(64, 128, 3, 1)
        self.conv6 = ConvLayer(128, 64, 1, 1)
        self.conv7 = ConvLayer(64, 128, 3, 1)
        self.conv9 = ConvLayer(128, 256, 3, 1)
        self.conv10 = ConvLayer(256, 128, 1, 1)
        self.conv11 = ConvLayer(128, 256, 3, 1)
        self.conv12 = ConvLayer(256, 512, 3, 1)
        self.conv13 = ConvLayer(512, 256, 1, 1)
        self.conv14 = ConvLayer(256, 512, 3, 1)
        self.conv15 = ConvLayer(512, 30, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2) #2
        x = self.conv3(x)
        x = F.max_pool2d(x, 2, 2) #4
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = F.max_pool2d(x, 2, 2) #8
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)

        return x

if __name__  == "__main__":
    #net = FastYOLO()
    #summary(net, (3, 416, 416))

    net = CharacterSeg()
    summary(net, (3, 240, 80))
    #print(net)