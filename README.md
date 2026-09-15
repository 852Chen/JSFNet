## Source Code
Complete source code package:
- Baidu Netdisk: [JSFNet源码.zip](https://pan.baidu.com/s/1Yt_63RxvGjS3oOVARO30eg)
- Access code: `dv46`

> ⚠️ Note: This compressed package contains source code, configuration files, **and pre‑processed dataset split/index files required for reproducing experimental results**. Raw public datasets (CamVid, PASCAL VOC2012) need to be downloaded from their official websites separately.
##Dataset
Pre-processed CamVid dataset (11 classes):

Baidu Netdisk: camvid11.zip
https://pan.baidu.com/s/1ongIUICCqikNWdq4yrw0BA?pwd=vnag
Access code: vnag

## Environment Requirements
- Python >= 3.8
- PyTorch >=1.10
- torchvision
- opencv‑python
- numpy

## Files Description
- `deeplab.py`: network definition of JSFNet
- `train.py`: training script
- `predict.py`: inference / prediction script
- `get_miou.py`: mIoU evaluation script
- `summary.py`: model parameter & FLOPs statistics
- `voc_annotation.py`: dataset pre‑processing for PASCAL VOC2012
- Dataset split folders: pre‑processed index / split files for CamVid, PASCAL VOC2012 (included in the zip archive)
