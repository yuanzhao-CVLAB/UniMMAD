

from mymodels.RD_de_resnet import BasicBlock as UpBasicBlock
from mymodels.RD_resnet import BasicBlock as DownBasicBlock
# from mymodels.rd_multimodal import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from mymodels.norm_and_act import LayerNorm2d,norm_act
from mymodels.RD_resnet import  wide_resnet50_2
from mymodels.RD_de_resnet import  de_wide_resnet50_2
from mymodels.norm_and_act import conv1x1,conv3x3


class DepthwiseSeparable(nn.Module):
    def __init__(self, C: int, OC, K: int, norm="gn"):
        super().__init__()
        self.dw = nn.Conv2d(C, C, K, padding=K // 2, groups=C, bias=False)
        self.pw = nn.Conv2d(C, OC, 1, bias=False)
        self.na = norm_act(OC, norm)

    def forward(self, x):
        return self.na(self.pw(self.dw(x)))

class BottleNeckK(nn.Module):


    def __init__(self, K: int, C: int, norm="gn"):
        super().__init__()
        self.b1 = DepthwiseSeparable(C, C // 3, K, norm)
        self.b2 = DepthwiseSeparable(C, C // 3, K, norm)
        self.b3 = DepthwiseSeparable(C, C // 3, K, norm)
        # 产生三路注意力权重（全局）

        self.post = nn.Sequential(nn.Conv2d((C // 3) * 3, C, kernel_size=2, stride=2), norm_act(C, norm))

    def forward(self, x1, x2, x3):
        B, C, H2, W2 = x3.shape
        if x1.shape[-2:] != (H2, W2):
            x1 = F.interpolate(x1, size=(H2, W2), mode="bilinear", align_corners=False)
        if x2.shape[-2:] != (H2, W2):
            x2 = F.interpolate(x2, size=(H2, W2), mode="bilinear", align_corners=False)

        z1 = self.b1(x1)
        z2 = self.b2(x2)
        z3 = self.b3(x3)
        out = torch.cat([z1, z2, z3], dim=1)
        return self.post(out)


class OutFuse3(nn.Module):
    def __init__(self, C: int, OC, norm="gn"):
        super().__init__()
        self.proj = nn.Sequential(conv1x1(3 * C, C), norm_act(C, norm))
        self.head = nn.Sequential(nn.Conv2d(C, OC, kernel_size=2, stride=2, bias=False), norm_act(OC, norm))
        # 也可以用软注意力，这里用 concat 后 1×1，与你原“合并→卷积”逻辑一致

    def forward(self, b1, b3, b5):
        B, C, H, W = b1.shape
        if b3.shape[-2:] != (H, W):
            b3 = F.interpolate(b3, size=(H, W), mode="bilinear", align_corners=False)
        if b5.shape[-2:] != (H, W):
            b5 = F.interpolate(b5, size=(H, W), mode="bilinear", align_corners=False)
        x = torch.cat([b1, b3, b5], dim=1)
        x = self.proj(x)
        return self.head(x)


class MultiScaleFusionWithEmbedding(nn.Module):
    def __init__(self, channels=[256, 512, 1024], norm: str = "gn", ):
        super(MultiScaleFusionWithEmbedding, self).__init__()
        _, bn = wide_resnet50_2()
        E = 512
        self.decoder = de_wide_resnet50_2()
        # channels = [192,384,768]
        # self.conv_concat = bn.bn_layer
        self.conv_x1 = nn.Sequential(
            DownBasicBlock(channels[0], E, stride=2, groups=1, downsample=conv1x1(channels[0], E, stride=2)))
        self.conv_x2 = nn.Sequential(
            DownBasicBlock(channels[1], E, stride=1, groups=1, downsample=conv1x1(channels[1], E, stride=1)))
        self.conv_x3 = nn.Sequential(UpBasicBlock(channels[2], E, stride=2, groups=1,
                                                  upsample=nn.ConvTranspose2d(channels[2], E, kernel_size=4, stride=2,
                                                                              padding=1)))


        # ---- 三个 K 分支（接口不变：bnK(x1,x2,x3)）----
        self.bn1 = BottleNeckK(K=1, C=E, norm=norm)
        self.bn3 = BottleNeckK(K=3, C=E, norm=norm)
        self.bn5 = BottleNeckK(K=5, C=E, norm=norm)

        # ---- 融合与解码（接口不变）----
        self.out_bn3 = OutFuse3(C=E, OC=2048, norm=norm)
        self.final = nn.Sequential(
            conv1x1(256, channels[0], stride=1),
            conv1x1(512, channels[1], stride=1),
            conv1x1(1024, channels[2], stride=1),

        )

    def forward(self, share_out, ):
        x1, x2, x3 = share_out

        x1 = self.conv_x1(x1)
        x2 = self.conv_x2(x2)
        x3 = self.conv_x3(x3)
        b1 = self.bn1(x1, x2, x3)
        b3 = self.bn3(x1, x2, x3)
        b5 = self.bn5(x1, x2, x3)
        out = self.out_bn3(b1, b3, b5)
        x = self.decoder(out)

        x = [m(d) for m, d in zip(self.final, x)]
        loss = torch.tensor(0.)
        return x, loss