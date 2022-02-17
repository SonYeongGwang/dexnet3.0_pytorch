# Dexnet3.0 Pytorch
This repository is implementations of both train and prediction of Grasp Quality CNN (GQ-CNN) with Dexnet3.0 dataset using Pytorch modules.

For more information, please visit original project website and the paper
- [Project Website](https://berkeleyautomation.github.io/dex-net/)
- **Paper**: Dex-Net 3.0: Computing Robust Robot Suction Grasp Targets using a New Analytic Model and Deep Learning,
Mahler et al., ICRA 2018

This repository features:
* **dests.py** - Script for prefetching dexnet3.0 dataset onto RAM, split train/validation sets and more.
* **model.py** - Grasp Quality CNN model consists of torch.nn module.
* **training.py** - Run this script for training your model. The default options are as below.
  * The number of images: number_of_files x 1000 = 2,760,000 images.
  * Learning rate: 0.001
  * Momemtum: 0.99
  * Epochs : 25
  * Batch size: 64
  
* **predict.py** - Prediction with the trained GQ-CNN.

To fetch full dexnet3.0 dataset, there should be at least 20GB free space on your RAM.
If you want to train with a fewer images, reduce 'number_of_files' in the [training.py](https://github.com/SonYeongGwang/dexnet3.0_pytorch/blob/master/training.py#L91)

#### Dataset
- You can download dexnet3.0 dataset in the project website [dexnet3.0 dataset](https://berkeley.app.box.com/s/6mnb2bzi5zfa7qpwyn7uq5atb7vbztng/folder/38455887072)

#### Results
- Using GQ-CNN trained with the Dexnet3.0 dataset, stable suction grasping point on a object can be determined.

<div align="center">
<img src="https://github.com/SonYeongGwang/dexnet3.0_pytorch/blob/master/results/box_candidates.png" height="200px" width="340px">
<img src="https://github.com/SonYeongGwang/dexnet3.0_pytorch/blob/master/results/box_final.png" height="200px" width="340px">
<img src="https://github.com/SonYeongGwang/dexnet3.0_pytorch/blob/master/results/cylinder_candidates.png" height="200px" width="340px">
<img src="https://github.com/SonYeongGwang/dexnet3.0_pytorch/blob/master/results/cylinder_final.png" height="200px" width="340px">
<img src="https://github.com/SonYeongGwang/dexnet3.0_pytorch/blob/master/results/knife_candidates.png" height="200px" width="340px">
<img src="https://github.com/SonYeongGwang/dexnet3.0_pytorch/blob/master/results/knife_final.png" height="200px" width="340px">
</div>
