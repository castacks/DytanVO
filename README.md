# DytanVO: Joint Refinement of Visual Odometry and Motion Segmentation in Dynamic Environments

<p align="center">
    <a href="https://www.icra2023.org/"><img src="https://img.shields.io/badge/ICRA-2023-yellow"></a>
    <a href="https://arxiv.org/abs/2209.08430"><img src="https://img.shields.io/badge/arXiv-2209.08430-b31b1b"></a>
    <a href="https://youtu.be/6yO7RsZjSBQ"><img src="https://img.shields.io/badge/Video-Demo-critical?logo=youtube"></a>
    <a href="https://github.com/castacks/DytanVO/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg"></a>
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
We provide an environment file using [anaconda](https://www.anaconda.com/). The code has been tested on an RTX 2080Ti with CUDA 11.4.
```bash
conda env create -f environment.yaml
conda activate dytanvo
```

Compile [DCNv2](https://github.com/MatthewHowe/DCNv2).
```
cd Network/rigidmask/networks/DCNv2/; python setup.py install; cd -
```

## Models and Data
Coming soon

## Evaluation
Run inference on dynamic sequences in KITTI (loading the finetuned VO model at once)
```bash
traj=00_1
python -W ignore::UserWarning vo_trajectory_from_folder.py --vo-model-name vonet_ft.pkl  \
							   --seg-model-name segnet-kitti.pth  \
							   --kitti --kitti-intrinsics-file data/DynaKITTI/$traj/calib.txt  \
							   --test-dir data/DynaKITTI/$traj/image_2  \
							   --pose-file data/DynaKITTI/$traj/pose_left.txt 
```

Run inference on AirDOS-Shibuya (loading FlowNet and PoseNet separately)
```bash
traj=RoadCrossing03
python -W ignore::UserWarning vo_trajectory_from_folder.py --flow-model-name flownet.pkl  \
							   --pose-model-name posenet.pkl  \
							   --seg-model segnet-sf.pth  \
							   --airdos  \
							   --test-dir data/AirDOS_shibuya/$traj/image_0  \
							   --pose-file data/AirDOS_shibuya/$traj/gt_pose_quats.txt 
```

Running the above commands with the `--save-flow` tag, allows you to save intermediate optical flow outputs into the `results` folder.


Adjust the batch size and the worker number by `--batch-size 10`, `--worker-num 5`. 


## Acknowledgement
We built DytanVO on top of [TartanVO](https://github.com/castacks/tartanvo). We implemented the segmentation network by adapting [rigidmask](https://github.com/gengshan-y/rigidmask). We thank [Gengshan Yang](https://gengshan-y.github.io/) for his code and suggestions. 

## License
This software is BSD licensed.

Copyright (c) 2020, Carnegie Mellon University All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
