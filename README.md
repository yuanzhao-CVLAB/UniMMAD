<div align="center">
 
## UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection </br> via MoE-Driven Feature Decompression


### CVPR 2026 
[![arXiv](https://img.shields.io/badge/arXiv-2509.25934-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2509.25934) [![CVF](https://img.shields.io/badge/CVPR-Paper-b4c7e7.svg?style=plastic)](https://arxiv.org/abs/2509.25934)

</div>

PyTorch Implementation of CVPR 2026
"UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression". The first unified multi-modal and multi-class UAD model !!!

- **If you find this work useful, please star ⭐ the repo!**

- This project is released under **Apache-2.0 License**. 

## News
 
 - _02.2026_: Accepted by CVPR 2026🎉
 
 - _09.2025_: We have released the arXiv paper of UniMMAD v1, link:[https://arxiv.org/abs/2509.25934](https://arxiv.org/abs/2509.25934) .😛


## 🚀 TODO
- [x] Release arXiv paper.
- [ ] open training code
- [ ] open testing code
- [ ] release pre-trained UniMMAD Weights

## Introduction 
 UniMMAD, a unified framework for multi-modal and
multi-class anomaly detection. At the core of UniMMAD is
a Mixture-of-Experts (MoE)-driven feature decompression
mechanism, which enables adaptive and disentangled reconstruction tailored to specific domains.This process is guided
by a “general → specific” paradigm. In the encoding stage,
multi-modal inputs of varying combinations are compressed
into compact, general-purpose features. The encoder incorporates a feature compression module to suppress latent
anomalies, encourage cross-modal interaction, and avoid
shortcut learning. In the decoding stage, the general features
are decompressed into modality-specific and class-specific
forms via a sparsely-gated cross MoE, which dynamically
selects expert pathways based on input modality and class.
To further improve efficiency, we design a grouped dynamic
filtering mechanism and a MoE-in-MoE structure, reducing
parameter usage by 75% while maintaining sparse activation and fast inference. UniMMAD achieves state-of-the-art
performance on 9 anomaly detection datasets, spanning 3
fields, 12 modalities, and 66 classes. 

## MMAD Task

<div style="display: flex; justify-content: space-between;">
  <img src="imgs/MMAD_Task.png" alt="Image 1" style="width: 56%;"  />
  <img src="imgs/Datasets.png" alt="Image 2" style="width: 42%;"  />
</div>

Note:  (a) Existing methods rely on specialized models tailored to individual modalities and classes. (b) The proposed UniMMAD model unifies multi-modal and multi-class anomaly detection tasks within a single framework. (c) Visual examples, with modalities highlighted in white, class names in yellow, and anomaly regions marked by red boxes.  (d) Overview of the fields, modalities, and classes encompassed by UniMMAD.


## UniMMAD Framework
![UniMMAD](imgs/architecture.png)

## C-MoE Visualization
 <img src="imgs/Exp_CMoE.png" alt="Image 1" style="width: 50%;"  />

Note: Activation frequency of each MoE-leader across different datasets, illustrating domain-specific expert selection.


## Quantitative Comparison for Super-Multi-Class

| Methods             | MVTec-AD Image-level | MVTec-AD Pixel-level | VisA Image-level | VisA Pixel-level |
|----------------------|-----------------------|-----------------------|------------------|------------------|
| RD  | 95.8 / 97.8 / 95.0   | 95.1 / 51.5 / 90.7   | 88.4 / 89.9 / 86.7 | 96.8 /  38.7 / 87.0 |
| UniAD | 95.0 / 97.7 / 94.5   | 95.8 / 46.8 / 90.0   | 90.7 / 92.2 / 86.9 | 98.3 / 38.1 / 88.8 |
| ViTAD | 97.7 / 99.3 / 97.5   | 95.9 / 49.8 / 89.4   | 89.1 / 90.4 / 85.4 | 98.0 / 39.8 / 84.2 |
| MambaAD  | 94.7 / 97.7 / 94.5   | 96.3 / 51.5 / 90.5   | 90.2 / 91.4 / 87.5 | 97.7 / 39.4 / 87.9|
| INP-Former  | 99.2 / 99.5 / 98.6 | **98.2 / 60.7 / 93.8** | 95.2 / 95.7 / 91.9| 98.8 /  44.4 / **91.5**|
| **Ours**             | **99.4 / 99.6 / 98.7** | 98.1 / 60.2 / 93.0 | **95.5 / 96.2 / 92.4** | **98.9 / 47.2**  / 91.3|


Note:  Quantitative comparison on the MVTec-AD and VisA datasets under the super–multi-class setting with a resolution of $256 \times 256$, evaluated using image-level metrics ($\text{AUC}_I /\text{AP}_I / \text{MF1}_I$) and pixel-level metrics (  $\text{AUC}_P /\text{MF1}_P / \text{AUPRO}$).




## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yuanzhao-CVLAB/UniMMAD&type=Timeline)](https://www.star-history.com/#yuanzhao-CVLAB/UniMMAD&Timeline)****
