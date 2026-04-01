
import timm
from typing import Tuple, List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from mymodels.FCM import MultiScaleFusionWithEmbedding
from mymodels.cmoe import C_MoE




class Input_Preprocess(nn.Module):
    def __init__(self, args):
        super(Input_Preprocess, self).__init__()
        self.input_embedding = nn.Parameter(torch.randn((6, *args['img_size'])))
        self.stem = nn.Conv2d(6, 3, kernel_size=3, padding=1)

    def forward(self, share_img):
        input = self.input_embedding.unsqueeze(0).repeat((share_img.shape[0], 1, 1, 1))
        input[:, :share_img.shape[1]] += share_img
        share_out = self.stem(input)
        return share_out


class Pretrain_Encoder(nn.Module):
    def __init__(self, pretrain=True):
        super().__init__()


        self.model = model = timm.create_model(
            'wide_resnet50_2.tv2_in1k',
            pretrained=True,
            pretrained_cfg_overlay=dict(
                file="checkpoints/wide_resnet50_2.safetensors"),
            features_only=True,
        )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.model(x)  # .hidden_states
        out = out[1:-1]
        out = [self.act(o) for o in out]
        return out

    def get_channels(self):
        return [256, 512, 1024]

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)





# @TRAINER.register
class UniMMAD(object):
    def __init__(self, args):
        super(UniMMAD, self).__init__()

        self.piror_generator = Pretrain_Encoder()
        self.piror_generator = self.piror_generator.cuda()
        self.piror_generator.eval()
        channels = self.piror_generator.get_channels()

        self.c_moes = nn.Sequential(*([C_MoE(c) for c in channels]))
        self.general_multimodal_encoder = torch.nn.Sequential(Pretrain_Encoder(), MultiScaleFusionWithEmbedding(channels))
        self.input_embedding = Input_Preprocess(args)
    def train_step(self, batch, **kwargs):
        for modules in self.get_models()[1:]:
            if isinstance(modules, List):
                for m in modules:
                    m.train()
            else:
                modules.train()
        extra_loss, inputs, outputs = self.forward_step(batch, True, **kwargs)
        cos_loss = self.loss_fucntion(inputs, outputs, weight=batch["loss_weight"])
        loss = cos_loss + extra_loss
        return loss, {"Total_Loss": loss.item(), "extra_loss": extra_loss.item(), "cos_loss": cos_loss.item()}

    def cal_anomaly_map(self, ft_list, fs_list, out_size=[224, 224], uni_am=False, use_cos=True, amap_mode='add',
                        gaussian_sigma=0, weights=None):
        bs = ft_list[0].shape[0]
        a_map_list = []
        if uni_am:
            size = (ft_list[0].shape[2], ft_list[0].shape[3])
            for i in range(len(ft_list)):
                ft_list[i] = F.interpolate(F.normalize(ft_list[i], p=2), size=size, mode='bilinear', align_corners=True)
                fs_list[i] = F.interpolate(F.normalize(fs_list[i], p=2), size=size, mode='bilinear', align_corners=True)
            ft_map, fs_map = torch.cat(ft_list, dim=1), torch.cat(fs_list, dim=1)
            if use_cos:
                a_map = 1 - F.cosine_similarity(ft_map, fs_map, dim=1)
                a_map = a_map.unsqueeze(dim=1)
            else:
                a_map = torch.sqrt(torch.sum((ft_map - fs_map) ** 2, dim=1, keepdim=True))
            a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
            a_map = a_map.squeeze(dim=1).cpu().detach().numpy()
            anomaly_map = a_map
            a_map_list.append(a_map)
        else:
            for i in range(len(ft_list)):
                ft = ft_list[i]
                fs = fs_list[i]
                # ft_norm = F.normalize(ft, p=2)
                if use_cos:
                    a_map = 1 - F.cosine_similarity(ft, fs, dim=1)
                    a_map = a_map.unsqueeze(dim=1)
                else:
                    a_map = torch.sqrt(torch.sum((ft - fs) ** 2, dim=1, keepdim=True))
                a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
                a_map = a_map.squeeze(dim=1)
                a_map = a_map.cpu().detach().numpy()
                a_map_list.append(a_map)
            anomaly_map = self.intermodal_mean(np.stack(a_map_list))
        return anomaly_map

    def loss_fucntion(self, a, b, weight=None, gamma=2.0, reduction="mean"):
        cos_loss = torch.nn.CosineSimilarity(dim=1)
        loss = 0
        for item in range(len(a)):
            a_flat = a[item]
            b_flat = b[item]
            cosine_dist = 1 - cos_loss(a_flat, b_flat)
            focal_weight = cosine_dist.detach() ** gamma
            temp = cosine_dist * focal_weight  # focal loss
            if weight is not None:
                temp = temp.mean((-2, -1)) * weight
            if reduction == "mean":
                loss += torch.mean(temp)
            else:
                loss += temp
        return 3 * loss / len(a)

    def intermodal_mean(self, amap_np, eps=1e-8, k=2):
        return  np.sqrt((amap_np**2).sum(0,keepdims=True))


    def eval_step(self, batch, **kwargs):
        for modules in self.get_models()[1:]:
            if isinstance(modules, List):
                for m in modules:
                    m.eval()
            else:
                modules.eval()
        anomaly_mask = batch["anomaly_mask"]
        image = batch["specific_images"][:, :3]
        _, inputs, outputs = self.forward_step(batch, train=False)
        anomaly_map = self.cal_anomaly_map(inputs, outputs, [image.shape[-2], image.shape[-1]], uni_am=False,
                                           amap_mode='add', gaussian_sigma=4)
        out_mask_sm = torch.from_numpy(anomaly_map)
        return image, out_mask_sm
    def merge_feature(self,share_out,specimgs_out,B,cumindex):
        auxinp = []
        for layer in range(3):
            aux = [specimgs_out[layer][cumindex[i]:cumindex[i + 1]].mean(0, keepdim=True) for i in range(B)]
            auxinp.append(torch.cat(aux))
        share_out = [s + a for s, a in zip(share_out, auxinp)]
        return share_out
    def forward_step(self, batch, train=False, **kwargs):

        #################数据处理###############################
        share_img = batch["img"]
        specific_num = batch['modality_num']
        specific_imgs = torch.cat([batch['specific_images'][i, :specific_num[i]] for i in range(len(specific_num))], 0)
        B, C, H, W = share_img.shape

        #################模型推理###############################
        # 前向传播
        with torch.inference_mode():
            specimgs_out = self.piror_generator(specific_imgs)

        cumindex = [0] + torch.cumsum(specific_num, dim=0).tolist()

        share_out = self.input_embedding(share_img)
        enc = self.general_multimodal_encoder[0](share_out)
        #add inter-modal piror averages
        enc = self.merge_feature(enc,specimgs_out,B,cumindex)
        share_out, loss = self.general_multimodal_encoder[1](enc)
        share_out = [F.normalize(tensor, p=2, dim=1) for tensor in share_out]
        specimgs_out = [F.normalize(tensor, p=2, dim=1) for tensor in specimgs_out]



        ## layer-wise moe inference
        ## Same functionality as moe_inference function with faster training.
        def faster_moe_inference(C_MoE, layer_share_out, layer_spec_out, res_in, res_out):
            # 预计算每个模态的 batch 掩码和在 layer_spec_out 里的索引
            B = specific_num.shape[0]
            M = 4  # 最多 4 种模态
            # 每个样本在拼接的 specific_imgs 里的起始位置
            starts = torch.cumsum(specific_num, dim=0) - specific_num  # [B]
            masks = [(specific_num > m) for m in range(M)]
            idxs = [starts[masks[m]] + m for m in range(M)]
            counts = [int(mask.sum()) for mask in masks]  # [N0,N1,N2,N3]


            # 1) 打包：把各模态的 share/t 连接为一个大 batch
            gen_packed = torch.cat(
                [layer_share_out[masks[m]] for m in range(M) if counts[m] > 0], dim=0
            )
            u_packed = torch.cat(
                [layer_spec_out[idxs[m]] for m in range(M) if counts[m] > 0], dim=0
            )
            # 2) MoE 前向
            p_packed, loss = C_MoE(gen_packed, u_packed,temperature=1)
            total_loss.append(loss)

            # 3) 拆分为各模态子批
            non_zero_counts = [c for c in counts if c > 0]
            p_chunks = list(torch.split(p_packed, non_zero_counts, dim=0))
            u_chunks = list(torch.split(u_packed, non_zero_counts, dim=0))

            # 4) 回填：用布尔掩码直接赋值（无需 fill_back）
            cz = 0
            for m in range(M):
                if counts[m] == 0:
                    continue
                p_chunk = p_chunks[cz]
                u_chunk = u_chunks[cz]
                cz += 1

                # 为该模态构造 [B,C,H,W] 的容器，并把选中样本位赋值
                C, H, W = p_chunk.shape[1:]
                p_full = layer_share_out.new_zeros((B, C, H, W))
                u_full = layer_share_out.new_zeros((B, C, H, W))
                p_full[masks[m]] = p_chunk
                u_full[masks[m]] = u_chunk

                res_out.append(p_full)
                res_in.append(u_full)

        ## Better Readability. Same functionality as faster_moe_inference function
        def moe_inference(C_MoE, layer_share_out, layer_spec_out, res_in, res_out):
            B = layer_share_out.shape[0]
            for b in range(B):
                for specific_index in range(cumindex[b], cumindex[b + 1]):
                    gen_f = layer_share_out[b].unsqueeze(0)
                    prior_u_i = layer_spec_out[specific_index].unsqueeze(0)
                    p_i, moe_loss = C_MoE(gen_f, prior_u_i,
                                          temperature=1)
                    total_loss.append(moe_loss)
                    res_out.append(p_i)
                    res_in.append(prior_u_i)
        res_in, res_out, total_loss = [], [], []
        for layer in range(3):
            layer_spec_out = specimgs_out[layer]  # [sum(specific_num), C, H, W]
            layer_share_out = share_out[layer]  # [B, C, H, W]
            faster_moe_inference(self.c_moes[layer],layer_share_out, layer_spec_out,res_in,res_out)


        loss = torch.stack(total_loss).mean()
        loss = loss * map_value(kwargs.get("epoch", 1))
        return loss, res_in, res_out

    def get_models(self):
        trainable_layer = "self.general_multimodal_encoder,self.input_embedding,self.c_moes"
        return (trainable_layer.split(","),
                *eval(trainable_layer))


def map_value(value):
    #200epoch 
    return max(1 - value / 200, 0.001)





