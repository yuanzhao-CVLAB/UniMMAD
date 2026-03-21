

import torch
import torch.nn as nn

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """3x3 convolution with padding"""
    return torch.nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                         padding=dilation, groups=groups, bias=False, dilation=dilation),
                               nn.BatchNorm2d(out_planes), nn.GELU())
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
# -----------------------------
def _choose_gn_groups(C: int) -> int:
    # pick a divisor of C, prefer <=32
    for g in [32, 16, 8, 4, 2, 1]:
        if C % g == 0:
            return g
    return 1
def norm_act(C, kind="gn"):
    if kind == "ln2d":
        return nn.Sequential(LayerNorm2d(C), nn.ReLU(inplace=True))
    groups = _choose_gn_groups(C)
    return nn.Sequential(nn.GroupNorm(groups, C), nn.ReLU(inplace=True))



class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm on 2D feature maps (B,C,H,W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        # x: [B,C,H,W]
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat * self.weight + self.bias
