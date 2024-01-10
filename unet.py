""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch
import torch.nn as nn
from thop import profile


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# 最大池化+两层卷积
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):

        x1 = self.up(x1)
        if (x1.size()[:2] != x2.size()[2:]):
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DSCBlock(nn.Module):
    "Depthwise conv + Pointwise conv"

    def __init__(self, in_channels, out_channels):
        super(DSCBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class O_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(O_Block, self).__init__()
        self.conv1 = DSCBlock(in_channels, in_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv1(x)
        x = self.conv1(x)
        x += identity
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv_out(x)
        return x


class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


# Unet 模型
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # self.inc = DoubleConv(n_channels, 48)
        self.inc = Down(3, 48)
        self.down1 = Down(48, 96)
        self.down2 = Down(96, 192)
        self.down3 = Down(192, 384)
        factor = 2 if bilinear else 1
        self.down4 = Down(384, 768)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.agg1 = aggregation(48)
        self.rfb3 = RFB_modified(192, 48)
        self.rfb4 = RFB_modified(384, 48)
        self.rfb5 = RFB_modified(768, 48)

    def forward(self, x):
        x1 = self.inc(x)  # [1, 64, 512, 512]
        x2 = self.down1(x1)  # [1, 128, 256, 256]
        x3 = self.down2(x2)  # [1, 256, 128, 128]
        x4 = self.down3(x3)  # [1, 512, 64, 64]
        x5 = self.down4(x4)  # [1, 1024, 32, 32]

        x3_rfb = self.rfb3(x3)
        x4_rfb = self.rfb4(x4)
        x5_rfb = self.rfb5(x5)
        """UNet特征解码_wsy"""
        x_agg = self.agg1(x5_rfb, x4_rfb, x3_rfb)              #Deep Feature Aggreagtion Dercoder_wsy
        x_u_feat = F.interpolate(x_agg, scale_factor=8, mode='bilinear')
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        # print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)
        # logits = self.outc(x)
        return x2, x3, x4, x5, x_u_feat


if __name__ == '__main__':
    # img = torch.rand(8, 256, 224, 224)
    # model = UNet(3, 1, bilinear=False)
    # dummy_input = torch.randn(1, 3, 224, 224)
    # flops, params = profile(model, (dummy_input,))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
    # out1, out2, out3, out4, out5 = model(img)
    # print(out1.shape, out2.shape, out3.shape, out4.shape, out5.shape)

    model = RFB_modified(384, 48)
    dummy_input = torch.randn(8, 384, 14, 14)
    flops, params = profile(model, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
