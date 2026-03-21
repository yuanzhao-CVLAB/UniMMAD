import numpy as np
from tqdm import tqdm

import pandas as pd
import time
from tabulate import tabulate
from myutils.metric import cal_metric
from accelerate.logging import get_logger
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.mmad_datasets import get_dataset
import kornia.filters as K
from accelerate import Accelerator
import torch
device = 'cuda'

logger = get_logger(__name__)
# ① 生成 2-D 核
kernel2d = K.get_gaussian_kernel2d((15, 15), (4., 4.)).to(device)   # [15,15]
kernel2d = kernel2d.squeeze()
# ② 升维成 4-D: [1,1,15,15]
base_kernel = kernel2d[None, None, :, :]

def blur(x):
    """
    x: [B, C, H, W]  (CUDA)
    返回: 同形状张量，已经做过高斯模糊
    """
    B, C, H, W = x.shape

    # ③ 按通道 broadcast 到 [C,1,15,15]，不额外占显存
    kernel = base_kernel.expand(C, 1, -1, -1).to(x.device)

    # padding = kernel_size//2  (这里 15→7)
    return F.conv2d(x, kernel, padding=7, groups=C)


@torch.no_grad()
def evalute_single(accelerator: Accelerator,
                   dataloader,
                   args,
                   trainer,
                   subclass: str):

    # --------------- 预处理 ---------------
    device      = accelerator.device

    image_scores, image_labels = [], []
    pixel_scores, pixel_labels = [], []

    # --------------- 主循环 ---------------
    for batch in dataloader:

        target  = batch['has_anomaly']      # [B]
        gt_mask = batch['anomaly_mask']     # [B,1,h,w]

        _, pred_mask = trainer.eval_step(
            batch, output_dir=args["output_dir"], path=batch["path"][0]
        )                                   # pred_mask:[B,1,?,?]


        # GPU 高斯模糊
        pred_mask = blur(pred_mask)
        # ------- image-level 分数（top-k 均值） -------
        topk_vals = torch.topk(pred_mask.flatten(1), k=pred_mask.shape[-2]*pred_mask.shape[-1]//1000, dim=1).values  #
        img_score = topk_vals.mean(dim=1)         # [B]

        # ------- 收集 -------
        image_scores.append(img_score)
        image_labels.append(target)

        pixel_scores.append(pred_mask) # [B, H*W]
        pixel_labels.append(gt_mask)   # 同上

    # --------------- 聚合 ---------------
    image_scores = accelerator.gather_for_metrics(torch.cat(image_scores).to(device))
    image_labels = accelerator.gather_for_metrics(torch.cat(image_labels))

    pixel_scores = accelerator.gather_for_metrics(torch.cat(pixel_scores).to(device))
    pixel_labels = accelerator.gather_for_metrics(torch.cat(pixel_labels))

    res=cal_metric(image_labels,pixel_labels ,image_scores,pixel_scores,args["img_size"])
    return res


def evalute(trainer,args,accelerator):
    # if not  accelerator.is_main_process:return
    res = []
    classes =args['all_datasets_classes']# datasets_classes["mvtec"]
    bar = tqdm(classes,desc="evalutin ")
    for sub_class in bar:
        args['all_datasets_classes'] = sub_class
        bar.desc = f"evalutin class:{sub_class}"
        _, testing_dataset = get_dataset(args)
        test_loader = DataLoader(testing_dataset, batch_size=1, shuffle=False, num_workers=args["num_workers"])
        test_loader = accelerator.prepare(test_loader)
        res.append(evalute_single(accelerator,test_loader, args,trainer, sub_class))
    args['all_datasets_classes'] =classes
    logs = [[c] + list(r) for r, c in zip(res, classes)]
    res = np.array(res)

    logger.info("*" * 90)
    logger.info("*" * 90)
    logger.info("Results--------time:"+str(time.strftime('%Y-%m-%d %H:%M:%S')))
    

    class_avg = {}
    for c, r in zip(classes, res):
        data = class_avg.get(c.split("_")[-1], [])
        data.append(r)
        class_avg[c.split("_")[-1]] = data
    all_avg = []
    for key, val in class_avg.items():
        all_avg.append(np.array(val).mean(0).tolist())
        logs.append([f'{key}_avg'] + all_avg[-1])
    logs.append(["All_Class_Mean"] + np.array(res).mean(0).tolist())
    logs.append(["All_Datasets_Mean"] + np.array(all_avg).mean(0).tolist())

    col_names =  ["Objects", "I-Auroc", "I-AP", "I-F1max", "P-Auroc", "P-F1max", "AUPRO@30% ","P-AP",]
    pd_data = pd.DataFrame(logs, columns=col_names)
    logger.info("\n"+tabulate(pd_data.values, tablefmt="pipe"))

    return  res
