# DytanVO: Joint Refinement of Visual Odometry and Motion Segmentation in Dynamic Environments

<p align="center">
    <a href="https://www.icra2023.org/"><img src="https://img.shields.io/badge/ICRA-2023-red"></a>
    <a href="https://github.com/castacks/DytanVO/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg"></a>
    <a href="https://arxiv.org/abs/2209.08430"><img src="https://img.shields.io/badge/arXiv-2209.08430-b31b1b"></a>
    <a href="https://youtu.be/6yO7RsZjSBQ"><img src="https://img.shields.io/youtube/views/6yO7RsZjSBQ?style=social"></a>
</p>
<p align="center">
	DytanVO: Joint Refinement of Visual Odometry and Motion Segmentation in Dynamic Environments (ICRA 2023 under review)<br>
  By
  <a href="https://github.com/Geniussh/">Shihao Shen</a>, 
  <a href="http://missinglight.github.io/">Yilin Cai</a>, 
  <a href="http://www.wangwenshan.com/">Wenshan Wang</a>, and 
  <a href="https://theairlab.org/team/sebastian/">Sebastian Scherer</a>.
</p>

### What's new.

- 01-04-2023: Clean up and upload the codebase for DytanVO. Pretrained models and datasets are coming soon.

- 09-20-2022: Archive _Dynamic Dense RGB-D SLAM with Learning-Based Visual Odometry_, please check out the [legacy branch](https://github.com/Geniussh/DytanVO/tree/archived). The repo will be used to release codebase for the most recent ICRA 2023 submission.

- 05-15-2022: Release codebase for _Dynamic Dense RGB-D SLAM with Learning-Based Visual Odometry_.

## Introduction
DytanVO is a learning-based visual odometry (VO) based on its precursor, [TartanVO](https://github.com/castacks/tartanvo). It is the first supervised learning-based VO method that deals with dynamic environments. It takes two consecutive monocular frames in real-time and predicts camera ego-motion in an iterative fashion. It achieves an average improvement of 27.7% over state-of-the-art VO solutions in real-world dynamic environments, and even performs competitively among dynamic visual SLAM systems which optimize the trajectory on the backend. Experiments on plentiful unseen environments also demonstrate its generalizability.

## Installation
We provide an environment file using [anaconda](https://www.anaconda.com/). The code has been tested on an RTX 2080 with CUDA 11.4.
```bash
conda env create -f environment.yaml
conda activate dytanvo
```

Compile [DCNv2](https://github.com/MatthewHowe/DCNv2).
```
cd Network/rigidmask/networks/DCNv2/; python setup.py install; cd -
```
