"""Modular parts needed for unet definition

Similar to https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py"""
import torch.nn as nn
import torch
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
class single_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv(x)
        return x
    
class down(nn.Module):
    def __init__(self, in_ch, out_ch, conv_op):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_op(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, conv_op):
        super(inconv, self).__init__()
        self.conv = conv_op(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class up(nn.Module):
    def __init__(self, in_ch, out_ch, conv_op):
        super(up, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)#, padding=1)
        self.conv = conv_op(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

    
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
