# PyTorch-Human-object-relation-network
Unofficial Pytorch implementation of the work "Human-object Relation Network for Action Recognition in Still Images"
## Installation
This project is developed using **Python 3.1.2** and Pytorch 1.11.0.

### Python Packages
```txt
numpy==1.22.3
torch==1.11.0
torchaudio==0.11.0
torchvision==0.12.0
opencv-python
matplotlib==3.8.3
timm==0.5.4
tqdm
scipy
```
### Datasets

**VOC Action dataset**  
Please follow the dataset instructions provided in [Human-object relation network](https://github.com/WalterMa/Human-Object-Relation-Network?tab=readme-ov-file).

---

#### VOC 2012 dataset
1. Download the dataset and extract it to `~/data/`.  
2. Download the Scanpaths and BBoxes and extract them to `~/data/VOCdevkit/VOC2012/`.


### Training
**Training on VOC dataset**
```
python main.py
```
The model weights and log file will be saved in the ./models folder.

**Testing on VOC dataset**
```
python test.py
```

## Citation
```
@INPROCEEDINGS{horelation,
author={Wentao Ma and Shuang Liang},
booktitle={2020 IEEE International Conference on Multimedia and Expo (ICME)},
title={Human-Object Relation Network For Action Recognition In Still Images},
year={2020}}
```
## Disclaimer
The repository is an unofficial PyTorch re-implementation of the code from [Human-object relation network](https://github.com/WalterMa/Human-Object-Relation-Network?tab=readme-ov-file).
