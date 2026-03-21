import datetime
import glob
import os
import sys
# 设置环境变量
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['http_proxy'] = "http://127.0.0.1:7890"
os.environ['https_proxy'] = "http://127.0.0.1:7890"
from random import seed
import json
from collections import defaultdict
from accelerate import Accelerator
from accelerate.logging import get_logger
import torch.nn as nn
from torchvision.transforms import GaussianBlur

from myutils.loss import BinaryFocalLoss
from data.mmad_datasets import get_dataset,WeightedClassSampler,datasets_classes
import logging
logger = get_logger(__name__)
from train import  train
def defaultdict_from_json(jsonDict):
    func = lambda: defaultdict(str)
    dd = func()
    dd.update(jsonDict)
    return dd
def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="outputs/UniMMAD", required=False,
                        help="Directory to save outputs (checkpoints, logs).")
    parser.add_argument('--batchsize', type=int, default=10, required=False,
                        help="Batch size for training.")
    parser.add_argument('--resume', type=str, required=False,
                        default="none",
                        help="Path to checkpoint directory to resume training from.")
    parser.add_argument('--print_bar_step', type=int, default=10, required=False,
                        help="Interval (in epochs) to print progress and run evaluation.")
    parser.add_argument('--save_checkpoint_epoch', type=int, default=10, required=False,
                        help="Interval (in epochs) to save model checkpoints.")
    parser.add_argument('--num_workers', type=int, default=8, required=False,
                        help="Number of worker processes for data loading.")
    parser.add_argument('--lr', type=float, default=1e-3, required=False,
                        help="Learning rate.")
    parser.add_argument('--mode', type=str, default="eval", required=False,
                        help="Running mode: 'train' or 'eval'.")
    parser.add_argument('--all_datasets_classes', type=list, default=datasets_classes["unidatasets"], required=False,
                        help="List of dataset class names (default from datasets_classes['unidatasets']).")


    args = parser.parse_args()
    args_dict = vars(args)

    with open(f'./config.json', 'r') as f:
        args = json.load(f)
    config_dict = defaultdict_from_json(args)
    config_dict.update(args_dict)
    seed(42)
    os.makedirs(config_dict["output_dir"],exist_ok=True)
    os.makedirs(os.path.join(config_dict["output_dir"],"logs"),exist_ok=True)

    logger = set_logger(f'{config_dict["output_dir"]}/logs/{config_dict["data_type"]}_{datetime.datetime.now().strftime("%m_%d_%H_%M_%S_%f")[:-3]}.log')
    # 读取某个 python 文件并写入日志


    return logger,config_dict
def set_logger(log_file):


    # 配置根记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 创建一个文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    # 创建一个控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # 创建一个日志格式器，并将其添加到处理器中
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理器添加到根记录器中
    if not root_logger.hasHandlers():
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    return root_logger
if __name__ == '__main__':
    _,args = parse_args()
    loss_focal = BinaryFocalLoss(reduce=False)
    loss_smL1 = nn.SmoothL1Loss(reduction='none')
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision=args["amp"]
    )
    blur_func = GaussianBlur(kernel_size=3,sigma=(0.1,2.0))

    logger.info(json.dumps(args, indent=4))
    train(args, accelerator)

