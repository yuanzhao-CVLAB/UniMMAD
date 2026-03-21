
import torch.nn.functional as F
from torchvision import transforms

import warnings

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


import os
import random
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
import json
import numpy as np

datasets_classes = {
    "unidatasets":[



        #
        #
        # 'capsule_mvtec',
        # 'carpet_mvtec',
        # 'grid_mvtec',
        # 'leather_mvtec',
        # 'tile_mvtec',
        # 'wood_mvtec',
        # 'bottle_mvtec',
        # 'cable_mvtec',
        # 'hazelnut_mvtec',
        # 'metal_nut_mvtec',
        # 'pill_mvtec',
        # 'screw_mvtec',
        # 'toothbrush_mvtec',
        # 'transistor_mvtec',
        # 'zipper_mvtec',
        # # ###visa
        # 'candle_visa1cls',
        # 'capsules_visa1cls',
        # 'cashew_visa1cls',
        # 'chewinggum_visa1cls',
        # 'fryum_visa1cls',
        # 'macaroni1_visa1cls',
        # 'macaroni2_visa1cls',
        # 'pcb1_visa1cls',
        # 'pcb2_visa1cls',
        # 'pcb3_visa1cls',
        # 'pcb4_visa1cls',
        # 'pipe_fryum_visa1cls',


        #eyecandies
        'GummyBear_EyePre',
        'Lollipop_EyePre',  'Marshmallow_EyePre', 'LicoriceSandwich_EyePre',
        'ChocolatePraline_EyePre',
        'ChocolateCookie_EyePre',
        'PeppermintCandy_EyePre', 'HazelnutTruffle_EyePre',  'Confetto_EyePre','CandyCane_EyePre',
       # # # # # # # # # mvtec3d
       'bagel_mvtec3d',
         'cable_gland_mvtec3d',
         'carrot_mvtec3d', 'potato_mvtec3d', 'rope_mvtec3d',
        'foam_mvtec3d',
         'dowel_mvtec3d',
         'tire_mvtec3d',
        'peach_mvtec3d',
         'cookie_mvtec3d',
       #
       # #
       # # # # # # # # # # # # # # # # # # #
       # # # # # # # # # # # # # # # # # # # # # # #  # # #musen_ad
       'cotton_MulSen_AD',
       'cube_MulSen_AD',
        'zipper_MulSen_AD', 'toothbrush_MulSen_AD', 'spring_pad_MulSen_AD',
        'piggy_MulSen_AD', 'capsule_MulSen_AD',
        'light_MulSen_AD',
        'plastic_cylinder_MulSen_AD',
        'screen_MulSen_AD',
        'screw_MulSen_AD', 'flat_pad_MulSen_AD', 'nut_MulSen_AD',
         'button_cell_MulSen_AD', 'solar_panel_MulSen_AD',
       # # # # # # # # # # # # # # # # # # #
       # # # # # # # # # # # # # # # # # # #  #medicine
        'brats_BratsAD1K',
        'retinal_edema_RetinaAD',
        'liver_LiverAD',
       # # # # # # # # #  # 'brest_BUSIAD',
       'colon_HyperAD'
    ],
}

URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'


#                 ├── 000.png
def downsampling(x, size, to_tensor=False, bin=True):
    if to_tensor:
        x = torch.FloatTensor(x).to(c.device)
    down = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    if bin:
        down[down > 0] = 1
    return down


from torch.utils.data import Sampler
class WeightedClassSampler(Sampler):
    def __init__(self, dataset, num_samples=None):
        """
        自定义采样器，每个 epoch 重新按类别平衡采样数据。

        Args:
            dataset (UniDataset): 需要进行加权采样的数据集
            num_samples (int): 每个 epoch 采样的数据量（默认和 dataset 设定的相同）
        """
        self.dataset = dataset
        self.num_samples = num_samples if num_samples else dataset.sample_size
        self.indices = list(range(len(self.dataset)))  # 初始化索引

    def set_epoch_samples(self):
        """在每个 epoch 重新进行加权随机抽样"""
        self.indices = self.dataset.get_indices()  # 让数据集按照类别均衡采样
        # = list(range(len(self.dataset)))  # 重新计算索引，防止索引超出范围

    def __iter__(self):
        """返回新的索引"""
        return iter(self.indices)

    def __len__(self):
        """返回当前 epoch 的样本数"""
        return len(self.indices)

class UniDataset(Dataset):
    def __init__(self, cfg, train=True,image_transforms = None):
        super(UniDataset, self).__init__()
        self.train = train
        self.imgsize = cfg["img_size"][0]
        self.data_all = []
        self.root = cfg["root"]#
        meta_info = json.load(open(cfg["data_root"]))
        self.ref_meta_info = meta_info['train']
        self.meta_info = meta_info = meta_info['train' if self.train else 'test']
        self.cls_names = cfg["all_datasets_classes"]
        if not isinstance(self.cls_names, list):
            self.cls_names = [self.cls_names]
        self.cls_names = list(meta_info.keys()) if len(self.cls_names) == 0 else self.cls_names
        for cls_name in self.cls_names:
            self.data_all.extend(self.meta_info[cls_name])
        self.sample_size = self.total_size = len(self.data_all)
        random.shuffle(self.data_all) if self.train else None
        self.dataset_weight = cfg["dataset_weight"]
        
        if train:
            probabilities = []
            for data in self.data_all:
                cls = data['cls_name']
                pro = 1 / len(self.meta_info[cls])
                probabilities.append(pro)
            self.probabilities = torch.tensor(probabilities)
        else:
            self.probabilities = torch.ones((len(self.data_all)))
        self.image_transforms = image_transforms

    def get_indices(self):
        samples_indices = torch.multinomial(self.probabilities, num_samples=self.sample_size, replacement=True)
        return samples_indices

    def __len__(self):
        return len(self.data_all)

    def transform(self, x, img_len, binary=False):
        x = x.copy()
        x = torch.FloatTensor(x)
        if len(x.shape) == 2:
            x = x[None, None]
            channels = 1
        elif len(x.shape) == 3:
            x = x.permute(2, 0, 1)[None]
            channels = x.shape[1]
        else:
            raise Exception(f'invalid dimensions of x:{x.shape}')

        x = downsampling(x, (img_len, img_len), bin=binary)
        x = x.reshape(channels, img_len, img_len)
        return x

    def read_mask(self, mask_path):
        if not self.train and mask_path:
            with open(f"{self.root}/{mask_path}", 'rb') as f:
                mask = Image.open(f)
                mask = self.transform(np.array(mask), self.imgsize, binary=True)[:1]
                mask[mask > 0] = 1
        else:
            mask = torch.zeros((1, self.imgsize, self.imgsize))
        return mask

    def get_img(self, path, cls_name):
        if cls_name == "brats_BratsAD":
            flair = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
            t1 = cv2.cvtColor(cv2.imread(path.replace("flair", "t1")), cv2.COLOR_BGR2GRAY)
            t2 = cv2.cvtColor(cv2.imread(path.replace("flair", "t2")), cv2.COLOR_BGR2GRAY)
            return Image.fromarray(np.stack([flair, t1, t2], -1))
        else:
            return Image.open(path).convert('RGB')

    def read_img(self, path, typ):
        if typ.lower() == "rgb":
            img = Image.open(os.path.join(self.root, path)).convert('RGB')
            img = self.image_transforms(img)
            return img, img
        elif typ.lower() == "gray":
            img = Image.open(os.path.join(self.root, path)).convert('RGB')
            img = self.image_transforms(img)
            return img.mean(0, keepdim=True), img
        elif typ.lower() == "depth":
            sample = np.load(os.path.join(self.root, path))
            depth = sample[:, :, 0]
            fg = sample[:, :, -1]
            mean_fg = np.sum(fg * depth) / np.sum(fg)
            depth = fg * depth + (1 - fg) * mean_fg
            depth = (depth - mean_fg) * 100
            depth = self.transform(depth, self.imgsize, binary=False)
            return depth, depth.repeat(3, 1, 1)
        elif typ.lower() == "wofg_depth":
            sample = np.load(os.path.join(self.root, path))
            depth = sample[:, :, 0]
            mean_fg = depth.mean()
            depth = (depth - mean_fg) * 100
            depth = self.transform(depth, self.imgsize, binary=False)
            return depth, depth.repeat(3, 1, 1)
        else:
            raise Exception(f'invalid type:{typ}')


    def get_imgs(self, img_paths, img_types):
        share_img, specific_img = [], []
        for path, typ in zip(img_paths, img_types):
            share, spec = self.read_img(path, typ)
            share_img.append(share)
            specific_img.append(spec)
        return torch.cat(share_img, 0), specific_img

    def __getitem__(self, index):
        data = self.data_all[index]
        img_paths, img_types,depth_mean_std, mask_path, cls_name, specie_name, anomaly = (data['img_path'], data['img_type'],
                                                                                          data.get("depth_mean_std"), data['mask_path'],
                                                                                          data['cls_name'], data['specie_name'],
                                                                                          data['anomaly'])

        modality_index = torch.tensor([0,0,0,0])
        share_img_tensors = torch.zeros((4, self.imgsize, self.imgsize))

        share_img, specific_modality = self.get_imgs(img_paths, img_types)


        modality_num = len(specific_modality)

        spec_tensors = torch.zeros((4, 3, self.imgsize, self.imgsize))
        spec_tensors[:modality_num] = torch.stack(specific_modality, 0)
        modality_index[:modality_num] = torch.tensor( [{"rgb":1,"gray":2,"depth":3,"wofg_depth":4}[m.lower()] for m in img_types])
        mask = self.read_mask(mask_path)

        dataset = cls_name.split("_")[-1]
        loss_weight = self.dataset_weight[dataset]
        res = {
            "specific_images": spec_tensors,
            "modality_index": modality_index,
            "modality_num": modality_num,
            "img": share_img_tensors,
            'anomaly_mask': mask,
            'class_name': cls_name, 'has_anomaly': anomaly,
            'path': img_paths[0],
            'loss_weight':loss_weight,
        }

        return res
def get_dataset(cfg):
	norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
	image_transforms = transforms.Compose([transforms.Resize(cfg["img_size"]), transforms.ToTensor(),
                                                    transforms.Normalize(norm_mean, norm_std)], )
	train_set = UniDataset(cfg, train=True,image_transforms = image_transforms)
	test_set = UniDataset(cfg, train=False,image_transforms = image_transforms)
	return train_set, test_set
