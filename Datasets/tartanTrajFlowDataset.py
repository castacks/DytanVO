"""
# ==============================
# tartanTrajFlowDataset.py
# library for DytanVO data I/O
# Author: Wenshan Wang, Shihao Shen
# Date: 3rd Jan 2023
# ==============================
"""
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from os import listdir
from evaluator.transformation import pos_quats2SEs, pose2motion, SEs2ses
from .utils import make_intrinsics_layer

class TrajFolderDataset(Dataset):
    """scene flow synthetic dataset. """

    def __init__(self, imgfolder, transform = None, 
                    focalx = 320.0, focaly = 320.0, centerx = 320.0, centery = 240.0):
        
        files = listdir(imgfolder)
        self.rgbfiles = [(imgfolder +'/'+ ff) for ff in files if (ff.endswith('.png') or ff.endswith('.jpg'))]
        self.rgbfiles.sort()
        self.imgfolder = imgfolder

        print('Find {} image files in {}'.format(len(self.rgbfiles), imgfolder))

        self.N = len(self.rgbfiles) - 1

        # self.N = len(self.lines)
        self.transform = transform
        self.focalx = focalx
        self.focaly = focaly
        self.centerx = centerx
        self.centery = centery

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        imgfile1 = self.rgbfiles[idx].strip()
        imgfile2 = self.rgbfiles[idx+1].strip()
        img1 = cv2.imread(imgfile1)
        img2 = cv2.imread(imgfile2)

        res = {'img1': img1, 'img2': img2}

        h, w, _ = img1.shape
        intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
        res['intrinsic'] = intrinsicLayer

        if self.transform:
            res = self.transform(res)

        res['img1_raw'] = img1
        res['img2_raw'] = img2

        return res