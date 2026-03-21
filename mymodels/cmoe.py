import math
import random
import time

import timm
import torch
from typing import Tuple, List, Dict, Optional

from mymodels.RD_de_resnet import BasicBlock as UpBasicBlock
from mymodels.RD_resnet import BasicBlock as DownBasicBlock
# from mymodels.rd_multimodal import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import numpy as np
import cv2


from mymodels.norm_and_act import conv1x1,conv3x3,norm_act
from mymodels.FCM import MultiScaleFusionWithEmbedding


# -----------------------------
# C-MoE head (3 kernel sizes)
# -----------------------------
class C_MoE(nn.Module):
    """
    Build cross-condition gml, run three MiMoE branches (k=1,3,5) that share
    the same (share) features but with different base expert banks.
    """

    def __init__(self, ori_embed_dim: int, embed_dim: int = 256,
                 base_num: int = 8, router_num: int = 32,
                 top_k: int = 2, norm="gn"):
        super().__init__()
        self.embed_dim = embed_dim
        self.top_k = top_k

        # projectors
        self.conv_share = nn.Sequential(conv1x1(ori_embed_dim, embed_dim), norm_act(embed_dim, norm))
        self.conv_prior = nn.Sequential(conv1x1(ori_embed_dim, embed_dim), norm_act(embed_dim, norm))
        self.conv_cond = nn.Sequential(
            conv3x3(2 * embed_dim, embed_dim), norm_act(embed_dim, norm),
            conv1x1(embed_dim, embed_dim), norm_act(embed_dim, norm)
        )

        # 3 branches with different kernel sizes
        self.kernel_sizes = (1, 3, 5)
        self.MoEs = nn.ModuleList([
            MiMoE(embed_dim=embed_dim, base_num=base_num, router_num=router_num,
                  kernel_size=k, top_k=top_k, norm=norm)
            for k in self.kernel_sizes
        ])

        # base expert banks per kernel size
        self.expert_bases = nn.ParameterList([
            nn.Parameter(torch.randn(base_num, embed_dim, embed_dim, k, k) * 0.05)
            for k in self.kernel_sizes
        ])

        # fuse 3 branches (each returns C channels)
        self.conv_out_first = nn.Sequential(conv1x1(3 * embed_dim, ori_embed_dim), norm_act(ori_embed_dim, norm))
        self.conv_out_second = nn.Conv2d(embed_dim, 1, kernel_size=1, bias=True)  # anomaly head

        # prior head (for consistency / visualization)
        self.prior_head = nn.Conv2d(embed_dim, 1, kernel_size=1, bias=True)

        self.select_expert_dict = {}

    def forward(self, share: torch.Tensor, related: torch.Tensor,
                class_name: Optional[List[str]] = None,
                temperature: Optional[float] = None, batch=None):
        """
        share   : [B,Cs,H,W] from fused student backbone
        related : [B,Cr,H,W] from fused frozen backbone (or priors)
        """
        key = self.conv_share(share)  # [B,E,H,W]
        query = self.conv_prior(related)  # [B,E,H,W]
        cond = self.conv_cond(torch.cat([key, query], dim=1))
        gml = F.adaptive_avg_pool2d(cond, 1).flatten(1)  # [B,E]

        # run three kernel-size MoEs
        outs = []
        lb_losses = []
        gate_indices = []
        for i, (moe, base) in enumerate(zip(self.MoEs, self.expert_bases)):
            o, lb, idx, y = moe(key, gml, base, temperature=temperature)
            outs.append(o)
            lb_losses.append(lb)
            gate_indices.append(idx)

        feat = torch.cat(outs, dim=1)  # [B,3E,H,W]
        feat = self.conv_out_first(feat)  # [B,E,H,W]
        # anomaly_map = self.conv_out_second(feat)      # [B,1,H,W]
        # prior_map   = self.prior_head(query)          # [B,1,H,W]

        lb_loss = sum(lb_losses)
        extra = {
            "gate_indices": gate_indices,  # list of [B,K]
        }
        return feat, lb_loss

# -----------------------------
# Grouped parallel dynamic conv
# -----------------------------
def dynamic_conv_experts(values: torch.Tensor,
                         kernels: torch.Tensor,
                         padding: int) -> torch.Tensor:
    """
    values : [B, C, H, W]
    kernels: [B, E, C, C, k, k]   (E = K_route + 1; 0-th is fixed expert)
    return : [B, E, C, H, W]
    Single F.conv2d with groups=B*E for efficiency.
    """
    B, C, H, W = values.shape
    _, E, _, _, kH, kW = kernels.shape
    v = values.unsqueeze(1).expand(-1, E, -1, -1, -1)  # [B,E,C,H,W]
    v = v.reshape(1, B * E * C, H, W)
    w = kernels.reshape(B * E * C, C, kH, kW)
    y = F.conv2d(v, w, padding=padding, groups=B * E)
    y = y.view(B, E, C, H, W)
    return y


# -----------------------------
# Router (gate) with LB + z-loss
# -----------------------------
class NaiveGate(nn.Module):
    """
    Input: gml (global mixed latent) [B, embed_dim]
    Output:
        idx:   [B, top_k]        top-k indices
        score: [B, top_k]        normalized to sum=1
        lb:    scalar load-balance loss (cv^2 + z-loss)
    """

    def __init__(self, embed_dim: int, num_experts: int, top_k: int = 2,
                 temperature: float = 1.0, add_noise: bool = False,
                 z_loss_weight: float = 1e-4):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.temperature = temperature
        self.add_noise = add_noise
        self.z_loss_weight = z_loss_weight
        self.gate = nn.Linear(embed_dim, num_experts)

    @staticmethod
    def cv_squared(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # x: [E]
        mean = x.mean()
        var = x.var(unbiased=False)
        return var / (mean ** 2 + eps)

    def forward(self, gml: torch.Tensor):
        # gml: [B, C]
        logits = self.gate(gml)  # [B,E]
        if self.add_noise and self.training:
            noise = torch.randn_like(logits) * 0.01
            logits = logits + noise

        # z-loss (Switch-style) to stabilize large logits
        z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean() * self.z_loss_weight

        probs = F.softmax(logits / self.temperature, dim=-1)  # [B,E]
        # load-balance on probs (not logits)
        importance = probs.mean(dim=0)  # [E]
        lb = self.cv_squared(importance)

        score, idx = torch.topk(probs, k=self.top_k, dim=-1, largest=True, sorted=True)
        # normalize top-k scores to sum 1 (safer aggregation)
        score = score / (score.sum(dim=-1, keepdim=True) + 1e-8)

        return idx, score, (lb + z_loss)
def build_delta_kernel(C: int, k: int) -> torch.Tensor:
    """Return delta-like kernel [C,C,k,k], identity-ish conv."""
    w = torch.zeros(C, C, k, k)
    center = k // 2
    for c in range(min(C, C)):
        w[c, c, center, center] = 1.0
    return w

# -----------------------------
# MiMoE (MoE-in-MoE + GDF)
# -----------------------------
class MiMoE(nn.Module):
    """
    share: [B,C,H,W]  (values)
    gml:   [B,C]      (router input)
    expert_base: [N, C, C, k, k]  (shared base experts for this kernel size)
    """

    def __init__(self, embed_dim: int, base_num: int, router_num: int,
                 kernel_size: int, top_k: int = 2, norm="gn"):
        super().__init__()
        self.embed_dim = embed_dim
        self.base_num = base_num
        self.top_k = top_k
        self.kernel_size = kernel_size

        # fixed expert (shared knowledge) - initialized as delta kernel
        self.soft_expert = nn.Parameter(build_delta_kernel(embed_dim, kernel_size))

        # Router: choose "router experts" (each further mixes base experts)
        self.gate = NaiveGate(embed_dim=embed_dim, num_experts=router_num, top_k=top_k)

        # Per-router "leader" mixes base experts per channel:
        # experts bank: [router_num, base_num, embed_dim]
        self.experts = nn.Parameter(torch.randn(router_num, base_num, embed_dim) * 0.02)

        # lightweight pre/post
        self.pre = nn.Sequential(conv1x1(embed_dim, embed_dim), norm_act(embed_dim, norm))
        self.post = nn.Sequential(conv1x1(2 * embed_dim, embed_dim), norm_act(embed_dim, norm))

        # eval kernel cache
        self.register_buffer("_cache_valid", torch.tensor(0, dtype=torch.uint8), persistent=False)
        self._cached_idx = None
        self._cached_kernels = None

    def clear_cache(self):
        self._cache_valid.zero_()
        self._cached_idx = None
        self._cached_kernels = None

    def _compose_routed_kernels(self, idx: torch.Tensor, expert_base: torch.Tensor) -> torch.Tensor:
        """
        idx: [B,K] router expert indices
        expert_base: [N, C, C, k, k]
        return: [B,K,C,C,k,k]
        """
        B, K = idx.shape
        C = self.embed_dim
        # select per-sample the router experts weights: [B,K,base_num,embed_dim]
        sel = self.experts[idx]  # [B,K,N,C]
        sel = F.softmax(sel, dim=2)  # softmax on base_num dimension (MoE-in-MoE)

        # expert_base: [N, C, C, k, k]
        # Weighted sum over N -> [B,K,C,C,k,k]
        # einsum: (B K N C) x (N C Ci k k) -> (B K C Ci k k)
        routed_k = torch.einsum('bknc, ncihw -> bkcihw', sel, expert_base)
        return routed_k

    def forward(self, share: torch.Tensor, gml: torch.Tensor, expert_base: torch.Tensor,
                temperature: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        return:
            feat_out: [B,C,H,W]
            lb_loss : scalar
            idx     : [B,K] (for logging)
        """
        B, C, H, W = share.shape
        x = self.pre(share)

        # route
        if temperature is not None:
            old_temp = self.gate.temperature
            self.gate.temperature = temperature

        idx, score, lb = self.gate(gml)  # [B,K], [B,K], scalar

        if not self.training and self._cache_valid.item() == 1 and self._cached_idx is not None:
            if torch.equal(idx, self._cached_idx):
                routed_k = self._cached_kernels
            else:
                routed_k = self._compose_routed_kernels(idx, expert_base)
                self._cached_idx = idx.detach().clone()
                self._cached_kernels = routed_k.detach()
        else:
            routed_k = self._compose_routed_kernels(idx, expert_base)
            if not self.training:
                self._cached_idx = idx.detach().clone()
                self._cached_kernels = routed_k.detach()
                self._cache_valid.fill_(1)

        if temperature is not None:
            self.gate.temperature = old_temp

        # concat fixed expert kernel
        fixed_k = self.soft_expert.unsqueeze(0).unsqueeze(1).expand(B, 1, -1, -1, -1, -1)  # [B,1,C,C,k,k]
        kernels = torch.cat([fixed_k, routed_k], dim=1)  # [B,K+1,C,C,k,k]

        pad = self.kernel_size // 2
        y = dynamic_conv_experts(x, kernels, padding=pad)  # [B,K+1,C,H,W]
        fixed_out, routed_all = y[:, 0], y[:, 1:]  # [B,C,H,W], [B,K,C,H,W]
        routed_out = torch.einsum('bkchw,bk->bchw', routed_all, score)  # [B,C,H,W]

        out = torch.cat([fixed_out, routed_out], dim=1)  # [B,2C,H,W]
        out = self.post(out)  # [B,C,H,W]
        return out, lb, idx, y
