# WFN+CEL
This is an official PyTorch implementation of [Semi-supervised learning advances species recognition for aquatic biodiversity monitoring](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2024.1373755/full).

## Results

### FishNet
| #Metrics | Common | Medium | Rare | All |
|:---:|:---:|:---:|:---:|:---:|
| FixMatch+[CReST](https://ieeexplore.ieee.org/document/9578169) | 68.19 | 67.26 | 24.93 | 30.24 |
| FixMatch+[ABC](https://arxiv.org/abs/2110.10368) | 69.14 | 66.71 | 24.98 | 30.24 |
| FixMatch+[DARP](https://arxiv.org/abs/2007.08844) | 69.74 | 67.42 | 26.19 | 31.38 |
| FixMatch+[SAW](https://proceedings.mlr.press/v162/lai22b/lai22b.pdf) | 64.54 | 67.18 | 27.31 | 32.27 |
| FixMatch+[DASO](https://openaccess.thecvf.com/content/CVPR2022/papers/Oh_DASO_Distribution-Aware_Semantics-Oriented_Pseudo-Label_for_Imbalanced_Semi-Supervised_Learning_CVPR_2022_paper.pdf) | 65.74 | 67.70 | 27.07 | 32.13 |
| FixMatch+WFN+CEL | 69.58 | 68.36 | 32.61 | 37.11 |

## Usage

### Train
Train the model by 20% labeled data of [FishNet](https://openaccess.thecvf.com/content/ICCV2023/papers/Khan_FishNet_A_Large-scale_Dataset_and_Benchmark_for_Fish_Recognition_Detection_ICCV_2023_paper.pdf) dataset:

```
python train.py --ratio-labeled 0.2 --arch resnet50 --batch-size 12 --lr 0.0043 --warmup-epoch 0.05 --nesterov --use-ema --mu 4 --lambda-u 10 --T 0.4 --threshold 0.95 --seed 1024 --out results/fishnet_wfn_cel@0.2
```

Train the model by 10000 labeled data of CIFAR-100 dataset by using DistributedDataParallel:
```
python -m torch.distributed.launch --nproc_per_node 4 ./train.py --ratio-labeled 0.2 --arch resnet50 --batch-size 12 --lr 0.0043 --warmup-epoch 0.05 --nesterov --use-ema --mu 4 --lambda-u 10 --T 0.4 --threshold 0.95 --seed 1024 --out results/fishnet_wfn_cel@0.2
```

### Monitoring training progress
```
tensorboard --logdir=<your out_dir>
```

## Requirements
- python 3.6+
- torch 1.4
- torchvision 0.5
- tensorboard
- numpy
- tqdm
- apex (optional)

## References
- [Official TensorFlow implementation of FixMatch](https://github.com/google-research/fixmatch)
- [Unofficial PyTorch implementation of FixMatch](https://github.com/kekmodel/FixMatch-pytorch)
- [Unofficial PyTorch Reimplementation of RandAugment](https://github.com/ildoonet/pytorch-randaugment)
- [PyTorch FishNet models and datasets](https://fishnet-2023.github.io/)

## Citations
```
@article{ma2024semi,
  title={Semi-supervised learning advances species recognition for aquatic biodiversity monitoring},
  author={Ma, Dongliang and Wei, Jine and Zhu, Likai and Zhao, Fang and Wu, Hao and Chen, Xi and Li, Ye and Liu, Min},
  journal={Frontiers in Marine Science},
  volume={11},
  pages={1373755},
  year={2024},
  publisher={Frontiers Media SA}
}
```
