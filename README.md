# GAL-DeepLabv3plus


## Introduction
This is the official PyTorch implementation of [**Graph Attention Layer Evolves Semantic Segmentation for Road Pothole Detection: A Benchmark and Algorithms**](https://arxiv.org/abs/2109.02711), accepted by [IEEE T-IP](https://ieeexplore.ieee.org/document/9547682) for publication.

In this repo, we provide the training and testing setups on the [pothole dataset](http://gofile.me/4jm56/uE8P2xjbo). We have tested our code in Python 3.8.10, CUDA 11.1, and PyTorch 1.10.1.

<p align="center">
<img src="doc/GAL-DeepLabv3+.png" width="100%"/>
</p>

<p align="center">
<img src="doc/GAL.png" width="100%"/>
</p>


## Setup
Please setup the pothole dataset and the pretrained weight according to the following folder structure:
```
GAL-DeepLabv3plus
 |-- data
 |-- datasets
 |  |-- pothole
 |-- models
 |-- options
 |-- runs
 |  |-- tdisp_gal
 ...
```
The pothole dataset `datasets/pothole` can be downloaded from [here](http://gofile.me/4jm56/uE8P2xjbo), and the pretrained weight `runs/tdisp_gal` for our GAL-DeepLabv3+ can be downloaded from [here](http://gofile.me/4jm56/b2BYHNvpl).


## Usage

### Testing on the Pothole Dataset
For testing, please first setup the `runs/tdisp_gal` and the `datasets/pothole` folders as mentioned above. Then, run the following script:
```
bash ./scripts/test_gal.sh
```
for testing GAL-DeepLabv3+ with input transformed disparity images. The prediction results are stored in `testresults`.


### Training on the Pothole Dataset
For training, please first setup the `datasets/pothole` folder as mentioned above. Then, run the following script:
```
bash ./scripts/train_gal.sh
```
for training GAL-DeepLabv3+ with input transformed disparity images. The weights and the tensorboard record containing the loss curves as well as the performance on the validation set will be saved in `runs`.



## Citation
If you use this code for your research, please cite our paper.
```
@article{fan2021graph,
  title     = {Graph Attention Layer Evolves Semantic Segmentation for Road Pothole Detection: A Benchmark and Algorithms},
  author    = {Fan, Rui and Wang, Hengli and Wang, Yuan and Liu, Ming and Pitas, Ioannis},
  journal   = {IEEE Transactions on Image Processing},
  volume    = {30},
  number    = {},
  pages     = {8144-8154},
  year      = {2021},
  publisher = {IEEE},
  doi       = {10.1109/TIP.2021.3112316}
}
```


## Acknowledgement
Our code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [pytorch_segmentation](https://github.com/yassouali/pytorch_segmentation), [pytorch-deeplab-xception
](https://github.com/jfzhang95/pytorch-deeplab-xception), and [RTFNet](https://github.com/yuxiangsun/RTFNet).
