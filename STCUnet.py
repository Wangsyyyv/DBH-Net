# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging

import torch
import torch.nn as nn
import numpy as np
from unet import UNet
from Swin_Transform import SwinTransformerSys
import torch.nn.functional as F
from thop import profile

logger = logging.getLogger(__name__)


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class BiFusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(BiFusion_block, self).__init__()

        # channel attention for F_g, use SE Block
        self.fc1 = nn.Conv2d(ch_2, ch_2 // r_2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(ch_2 // r_2, ch_2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        # bi-linear modelling for both
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)

        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x):
        # bilinear pooling
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        bp = self.W(W_g * W_x)

        # spatial attention for cnn branch
        g_in = g
        g = self.compress(g)
        g = self.spatial(g)
        g = self.sigmoid(g) * g_in

        # channel attetion for transformer branch
        x_in = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * x_in
        fuse = self.residual(torch.cat([g, x, bp], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse


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
        # self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        # x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        # x = self.conv_res(x)
        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x = self.relu(x_cat + x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_planes, out_planes,
                                       kernel_size=kernel_size, stride=stride,
                                       padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Spatial_Attention(nn.Module):
    def __init__(self, kernel_size=3):
        super(Spatial_Attention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        y = self.sigmoid(y)
        return x * y


class Channel_Attention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Channel_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)  # 对应Squeeze操作
        y = y.view(b, c)
        y = self.fc(y)  # 对应Excitation操作
        y = y.view(b, c, 1, 1)
        y = y.expand_as(x)
        return x * y


class CABM_Block(nn.Module):
    def __init__(self, channel):
        super(CABM_Block, self).__init__()
        self.ca = Channel_Attention(channel)
        self.sa = Spatial_Attention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class SUFusionBlock(nn.Module):
    def __init__(self, in_c, out_c, l_c=0):
        super(SUFusionBlock, self).__init__()
        self.rfb = RFB_modified(in_c, in_c // 4)
        self.cabm = CABM_Block(in_c)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual = Residual(in_c * 3, out_c)
        self.conv1 = Conv(in_c * 2, in_c, 3, bn=True, relu=True)
        self.conv = Conv(in_c * 2 + l_c, in_c, 3, bn=True, relu=True)
        self.w_1 = Conv(in_c, in_c, 1, bn=True, relu=False)
        self.w_2 = Conv(in_c, in_c, 1, bn=True, relu=False)
        self.w = Conv(in_c, in_c, 3, bn=True, relu=True)

    def forward(self, u, s, l=None):
        u = self.rfb(u)
        s = self.rfb(s)
        if l is not None:
            l = self.maxpool(l)
            x = torch.cat([u, s, l], dim=1)
            return self.conv(x)
        #
        # u_w = self.w_1(u)
        # s_w = self.w_2(s)
        # b = self.w(u_w * s_w)
        #
        # u_c = self.cabm(u)
        # s_c = self.cabm(s)
        #
        # x = torch.cat([u_c, s_c, b], dim=1)
        #
        # x = self.residual(x)
        x = torch.cat([u, s], dim=1)
        return self.conv1(x)


class SUFusionBlock_2(nn.Module):
    def __init__(self, in_c, out_c, l_c=0):
        super(SUFusionBlock_2, self).__init__()
        self.rfb = RFB_modified(in_c, in_c // 4)
        #self.rfb1 = RFB_modified(in_c * 2, in_c // 4)
        self.cabm = CABM_Block(in_c)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual = Residual(in_c * 3, out_c)
        self.conv1 = Conv(in_c * 2, in_c, 3, bn=True, relu=True)
        self.conv = Conv(in_c * 2 + l_c, in_c, 3, bn=True, relu=True)
        self.w_1 = Conv(in_c, in_c, 1, bn=True, relu=False)
        self.w_2 = Conv(in_c, in_c, 1, bn=True, relu=False)
        self.w_3 = Conv(in_c // 2, in_c, 1, bn=True, relu=False)
        self.w = Conv(in_c, in_c, 3, bn=True, relu=True)

    def forward(self, u, s, l=None):
        #u = self.rfb(u)
        #s = self.rfb(s)
        if l is not None:
            l = self.maxpool(l)
            u_w = self.w_1(u)
            s_w = self.w_2(s)
            l_w = self.w_3(l)

            b = self.w(u_w * s_w * l_w)
            #u_c = self.cabm(u)
            #s_c = self.cabm(s)
            x = torch.cat([u, s, b], dim=1)
            x = self.residual(x)
            return x

        u_w = self.w_1(u)
        s_w = self.w_2(s)
        b = self.w(u_w * s_w)
        #u_c = self.cabm(u)
        #s_c = self.cabm(s)

        x = torch.cat([u, s, b], dim=1)

        x = self.residual(x)
        #x = torch.cat([u, s], dim=1)
        return x



class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x) + self.identity(x))


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_ch1=48, out_ch=48, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch1 + in_ch2, out_ch)

        if attn:
            self.attn_block = Attention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):

        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)


class STCUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=1, zero_head=False, drop_rate=0.2):
        super(STCUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
                                            patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                            in_chans=config.MODEL.SWIN.IN_CHANS,
                                            num_classes=self.num_classes,
                                            embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                            depths=config.MODEL.SWIN.DEPTHS,
                                            num_heads=config.MODEL.SWIN.NUM_HEADS,
                                            window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                            qk_scale=config.MODEL.SWIN.QK_SCALE,
                                            drop_rate=config.MODEL.DROP_RATE,
                                            drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                            ape=config.MODEL.SWIN.APE,
                                            patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                            use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        self.Unet = UNet(3, 1, bilinear=False)

        self.up_1 = SUFusionBlock_2(96, 96)
        self.up_2 = SUFusionBlock_2(192, 192, 96)
        self.up_3 = SUFusionBlock_2(384, 384, 192)
        self.up_4 = SUFusionBlock_2(768, 768, 384)

        self.cat_1 = Up(in_ch1=768, out_ch=384, in_ch2=384, attn=True)
        self.cat_2 = Up(in_ch1=384, out_ch=192, in_ch2=192, attn=True)
        self.cat_3 = Up(in_ch1=192, out_ch=96, in_ch2=96, attn=True)
        self.cat_4 = Up(in_ch1=96, out_ch=48, in_ch2=48, attn=True) # 没用到
        self.cat_5 = Up() # 没用到
        self.conv_out = nn.Conv2d(48, num_classes, kernel_size=1)

        self.final_1 = nn.Sequential(
            Conv(96, 48, 3, bn=True, relu=True),
            Conv(48, num_classes, 3, bn=False, relu=False)
        )

        self.final_2 = nn.Sequential(
            Conv(96, 48, 3, bn=True, relu=True),
            Conv(48, 48, 3, bn=True, relu=True),
            Conv(48, num_classes, 3, bn=False, relu=False)
        )

        self.final_3 = nn.Sequential(
            Conv(192, 48, 1, bn=True, relu=True),
            Conv(48, 48, 3, bn=True, relu=True),
            Conv(48, num_classes, 3, bn=False, relu=False)
        )

        self.drop = nn.Dropout2d(drop_rate)
        self.sup1 = Up(in_ch1=768, out_ch=384)
        self.sup2 = Up(in_ch1=384, out_ch=192)

    def forward(self, x):
        x_s_4, x_s_3, x_s_2, x_s_1, x_s_feat = self.swin_unet(x)

        x_u_2, x_u_3, x_u_4, x_u_5, x_u_feat = self.Unet(x)

        x_c_1 = self.up_1(x_u_2, x_s_1)
        x_c_2 = self.up_2(x_u_3, x_s_2, x_c_1)
        x_c_3 = self.up_3(x_u_4, x_s_3, x_c_2)
        x_c_4 = self.up_4(x_u_5, x_s_4, x_c_3)

        x_f_1 = self.cat_1(x_c_4, x_c_3)
        x_f_2 = self.cat_2(x_f_1, x_c_2)
        x_f_3 = self.cat_3(x_f_2, x_c_1)

        #x_f_4 = self.cat_4(x_f_3)
        x_f_4 = self.final_2(x_f_3)
        x_f_4 = F.interpolate(x_f_4, scale_factor=4, mode='bilinear')

        return x_f_4, x_s_feat, x_u_feat


    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        print(pretrained_path)
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")


