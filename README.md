# DytanVO: Joint Refinement of Visual Odometry and Motion Segmentation in Dynamic Environments

<p align="center">
    <a href="https://www.icra2023.org/"><img src="https://img.shields.io/badge/ICRA-2023-yellow?logo=ieee"></a>
    <a href="https://arxiv.org/abs/2209.08430"><img src="https://img.shields.io/badge/arXiv-2209.08430-b31b1b"></a>
    <a href="https://youtu.be/6yO7RsZjSBQ"><img src="https://img.shields.io/badge/Video-Demo-critical?logo=youtube"></a>
    <a href="https://github.com/castacks/DytanVO/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg"></a>
</p>
<p align="center">
	DytanVO: Joint Refinement of Visual Odometry and Motion Segmentation in Dynamic Environments (ICRA 2023)<br>
  By
  <a href="https://github.com/Geniussh/">Shihao Shen</a>, 
  <a href="http://missinglight.github.io/">Yilin Cai</a>, 
  <a href="http://www.wangwenshan.com/">Wenshan Wang</a>, and 
  <a href="https://theairlab.org/team/sebastian/">Sebastian Scherer</a>.
</p>

### What's new.

- 01-17-2023: Our paper has been accepted to ICRA 2023!

- 01-05-2023: Clean up and upload the codebase for _DytanVO_. Pretrained weights and datasets are also ready.

- 09-20-2022: Remove _Dynamic Dense RGB-D SLAM with Learning-Based Visual Odometry_. The repo will be used to release codebase for the most recent ICRA 2023 submission.


## Introduction
DytanVO is a learning-based visual odometry (VO) based on its precursor, [TartanVO](https://github.com/castacks/tartanvo). It is the first supervised learning-based VO method that deals with dynamic environments. It takes two consecutive monocular frames in real-time and predicts camera ego-motion in an iterative fashion. It achieves an average improvement of 27.7% over state-of-the-art VO solutions in real-world dynamic environments, and even performs competitively among dynamic visual SLAM systems which optimize the trajectory on the backend. Experiments on plentiful unseen environments also demonstrate its generalizability.


## Installation
We provide an environment file using [anaconda](https://www.anaconda.com/). The code has been tested on an RTX 2080Ti with CUDA 11.4.
```bash
conda env create -f environment.yml
conda activate dytanvo
```

Compile [DCNv2](https://github.com/MatthewHowe/DCNv2).
```
cd Network/rigidmask/networks/DCNv2/; python setup.py install; cd -
```


## Models and Data

### Pretrained weights
Download [here](https://drive.google.com/file/d/1ujYmKv5FHXYe1KETabTnSs-R2OE0KJV3/view?usp=share_link) and unzip it to the `models` folder. 

### KITTI dynamic sequences
Original sequences in [KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) are trimmed into sub-sequences which contain moving pedestrians, vehicles and cyclists so that VO's robustness to dynamic objects can be explicitly evaluated. Download [DynaKITTI](https://drive.google.com/file/d/1BDnraRWzNf938UsfprWIkcqCSfOUyGt9/view?usp=share_link) and unzip it to the `data` folder. Please cite this paper if you find it useful in your work. 

### AirDOS-Shibuya
Follow [tartanair-shibuya](https://github.com/haleqiu/tartanair-shibuya) and download it to the `data` folder.

### (Optional) Scene Flow
One can also test the model on [Scene Flow datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), which was used to train both the VO and the segmentation networks. Scene Flow datasets have very challenging sequences with large areas of dynamic objects in image frames. 

You can create symbolic links to wherever the datasets were downloaded in the `data` folder.

```Shell
├── data
    ├── AirDOS_shibuya
        ├── RoadCrossing03
            ├── image_0
            ├── ...
            ├── gt_pose.txt
        ├── RoadCrossing04
        ├── ...
    ├── DynaKITTI
        ├── 00_1
            ├── image_2
            ├── ...
            ├── pose_left.txt
            ├── calib.txt
        ├── 01_0
        ├── ...
    ├── SceneFlow
        ├── FlyThings3D
            ├── frames_cleanpass
            ├── frames_finalpass
            ├── optical_flow
            ├── camera_data
        ├── Driving
        ├── Monkaa
    ├── ...
```


## Evaluation
Create a folder to save output flow, segmentation, or poses. 
```bash
mkdir results
```

### Dynamic sequences in KITTI (loading the finetuned VO model at once)
```bash
traj=00_1
python -W ignore::UserWarning vo_trajectory_from_folder.py --vo-model-name vonet_ft.pkl  \
							   --seg-model-name segnet-kitti.pth  \
							   --kitti --kitti-intrinsics-file data/DynaKITTI/$traj/calib.txt  \
							   --test-dir data/DynaKITTI/$traj/image_2  \
							   --pose-file data/DynaKITTI/$traj/pose_left.txt 
```

### AirDOS-Shibuya (loading FlowNet and PoseNet separately)
```bash
traj=RoadCrossing03
python -W ignore::UserWarning vo_trajectory_from_folder.py --flow-model-name flownet.pkl  \
							   --pose-model-name posenet.pkl  \
							   --seg-model segnet-sf.pth  \
							   --airdos  \
							   --test-dir data/AirDOS_shibuya/$traj/image_0  \
							   --pose-file data/AirDOS_shibuya/$traj/gt_pose.txt 
```

### Scene Flow
```bash
img=Driving/frames_finalpass/15mm_focallength/scene_forwards/fast/left
pose=Driving/camera_data/15mm_focallength/scene_forwards/fast/camera_data.txt
python -W ignore::UserWarning vo_trajectory_from_folder.py --flow-model-name flownet.pkl  \
							   --pose-model-name posenet.pkl  \
							   --seg-model segnet-sf.pth  \
							   --sceneflow  \
							   --test-dir data/SceneFlow/$img  \
							   --pose-file data/SceneFlow/$pose
```

Add `--save-flow` tag to save intermediate optical flow outputs into the `results` folder.

Adjust the batch size and the worker number by `--batch-size 10`, `--worker-num 5`. 


## (Optional) Segmentation Mask Ground Truth
If your dataset has ground truth for camera motion, optical flow and disparity change across consecutive frames, we provide an example script to automatically generate ground truth of segmentation mask given these two modalities based on the pure geometry for the [Scene Flow datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html). 

```bash
python Datasets/segmask_gt.py --database data/SceneFlow --frames_pass clean --dataset FlyingThings3D
```

Add `--debug` flag to save visualizations of the generated masks.

## Citation
If you find our code, paper or dataset useful, please cite
```bibtex
@article{shen2022dytanvo,
  title={DytanVO: Joint Refinement of Visual Odometry and Motion Segmentation in Dynamic Environments},
  author={Shen, Shihao and Cai, Yilin and Wang, Wenshan and Scherer, Sebastian},
  journal={arXiv preprint arXiv:2209.08430},
  year={2022}
}
```

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
