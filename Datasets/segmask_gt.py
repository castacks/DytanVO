"""
# ==============================
# segmask_gt.py
# library to generate groundtruth 
# segmentation mask given flow and
# disparity change
# (Adapted from code for rigidmask)
# Author: Shihao Shen
# Date: 14th Sep 2022
# ==============================
"""

import argparse
import os
import os.path
import glob
import numpy as np
import cv2
from PIL import Image
from flowlib import read_flow, readPFM, flow_to_image

def dataloader(filepath, fpass='frames_cleanpass', level=6):
    iml0 = []
    iml1 = []
    flowl0 = []
    disp0 = []
    dispc = []
    calib = []
    level_stars = '/*'*level
    candidate_pool = glob.glob('%s/optical_flow%s'%(filepath,level_stars))
    for flow_path in sorted(candidate_pool):
        # if 'TEST' in flow_path: continue
        if 'flower_storm_x2/into_future/right/OpticalFlowIntoFuture_0023_R.pfm' in flow_path:
            print('Skipping %s' % flow_path)
            continue
        if 'flower_storm_x2/into_future/left/OpticalFlowIntoFuture_0023_L.pfm' in flow_path:
            print('Skipping %s' % flow_path)
            continue
        if 'flower_storm_augmented0_x2/into_future/right/OpticalFlowIntoFuture_0023_R.pfm' in flow_path:
            print('Skipping %s' % flow_path)
            continue
        if 'flower_storm_augmented0_x2/into_future/left/OpticalFlowIntoFuture_0023_L.pfm' in flow_path:
            print('Skipping %s' % flow_path)
            continue
        # if 'FlyingThings' in flow_path and '_0014_' in flow_path:
        #     print('Skipping %s' % flow_path)
        #     continue
        # if 'FlyingThings' in flow_path and '_0015_' in flow_path:
        #     print('Skipping %s' % flow_path)
        #     continue
        idd = flow_path.split('/')[-1].split('_')[-2]
        if 'into_future' in flow_path:
            idd_p1 = '%04d'%(int(idd)+1)
        else:
            idd_p1 = '%04d'%(int(idd)-1)
        if os.path.exists(flow_path.replace(idd,idd_p1)): 
            d0_path = flow_path.replace('/into_future/','/').replace('/into_past/','/').replace('optical_flow','disparity')
            d0_path = '%s/%s.pfm'%(d0_path.rsplit('/',1)[0],idd)
            dc_path = flow_path.replace('optical_flow','disparity_change')
            dc_path = '%s/%s.pfm'%(dc_path.rsplit('/',1)[0],idd)
            im_path = flow_path.replace('/into_future/','/').replace('/into_past/','/').replace('optical_flow',fpass)
            im0_path = '%s/%s.png'%(im_path.rsplit('/',1)[0],idd)
            im1_path = '%s/%s.png'%(im_path.rsplit('/',1)[0],idd_p1)

            # This will skip any sequence that contains less than 10 poses in camera_data.txt
            with open('%s/camera_data.txt'%(im0_path.replace(fpass,'camera_data').rsplit('/',2)[0]),'r') as f:
               if 'FlyingThings' in flow_path and len(f.readlines())!=40: 
                   print('Skipping %s' % flow_path)
                   continue

            iml0.append(im0_path)
            iml1.append(im1_path)
            flowl0.append(flow_path)
            disp0.append(d0_path)
            dispc.append(dc_path)
            calib.append('%s/camera_data.txt'%(im0_path.replace(fpass,'camera_data').rsplit('/',2)[0]))
    return iml0, iml1, flowl0, disp0, dispc, calib

def default_loader(path):
    return Image.open(path).convert('RGB')

def flow_loader(path):
    if '.pfm' in path:
        data =  readPFM(path)[0]
        data[:,:,2] = 1
        return data
    else:
        return read_flow(path)

def load_exts(cam_file):
    with open(cam_file, 'r') as f:
        lines = f.readlines()

    l_exts = []
    r_exts = []
    for l in lines:
        if 'L ' in l:
            l_exts.append(np.asarray([float(i) for i in l[2:].strip().split(' ')]).reshape(4,4))
        if 'R ' in l:
            r_exts.append(np.asarray([float(i) for i in l[2:].strip().split(' ')]).reshape(4,4))
    return l_exts,r_exts        

def disparity_loader(path):
    if '.png' in path:
        data = Image.open(path)
        data = np.ascontiguousarray(data,dtype=np.float32)/256
        return data
    else:    
        return readPFM(path)[0]

# triangulation
def triangulation(disp, xcoord, ycoord, bl=1, fl = 450, cx = 479.5, cy = 269.5):
    depth = bl*fl / disp # 450px->15mm focal length
    X = (xcoord - cx) * depth / fl
    Y = (ycoord - cy) * depth / fl
    Z = depth
    P = np.concatenate((X[np.newaxis],Y[np.newaxis],Z[np.newaxis]),0).reshape(3,-1)
    P = np.concatenate((P,np.ones((1,P.shape[-1]))),0)
    return P

def exp_loader(index, iml0s, iml1s, flowl0s, disp0s=None, dispcs=None, calibs=None):
    '''
    index: index of the frame in the file lists below
    iml0s: a file list of the first frames
    iml1s: a file list of the second frames
    flowl0s: a file list of the optical w.r.t. iml0s
    disp0s: a file list of the disparity w.r.t. iml0s
    dispcs: a file list of the disparity change w.r.t. disp0s
    calibs: a file list of the camera extrinsics
    '''
    iml0 = iml0s[index]
    iml1 = iml1s[index]
    flowl0 = flowl0s[index]
    
    iml0 = default_loader(iml0)
    iml1 = default_loader(iml1)

    flowl0 = flow_loader(flowl0)
    flowl0[:,:,-1][flowl0[:,:,0] == np.inf] = 0 
    flowl0[:,:,0][~flowl0[:,:,2].astype(bool)] = 0
    flowl0[:,:,1][~flowl0[:,:,2].astype(bool)] = 0
    flowl0 = np.ascontiguousarray(flowl0, dtype=np.float32)
    flowl0[np.isnan(flowl0)] = 1e6
    
    bl = 1
    if '15mm_' in calibs[index]: 
        fl = 450
    else:
        fl = 1050
    cx = 479.5
    cy = 269.5
    intr = [[fl],[cx],[cy],[bl],[1],[0],[0],[1],[0],[0]]

    d1 = np.abs(disparity_loader(disp0s[index]))
    d2 = np.abs(disparity_loader(dispcs[index]) + d1)
    
    flowl0[:,:,2] = np.logical_and(np.logical_and(flowl0[:,:,2] == 1, d1 != 0), d2 != 0).astype(float)
    
    shape = d1.shape
    mesh = np.meshgrid(range(shape[1]), range(shape[0]))
    xcoord = mesh[0].astype(float)
    ycoord = mesh[1].astype(float)

    # triangulation in two frames
    P0 = triangulation(d1, xcoord, ycoord, bl=bl, fl=fl, cx=cx, cy=cy)
    P1 = triangulation(d2, xcoord + flowl0[:,:,0], ycoord + flowl0[:,:,1], bl=bl, fl=fl, cx=cx, cy=cy)
    depth0 = P0[2]
    depth1 = P1[2]

    depth0 = depth0.reshape(shape).astype(np.float32)
    flow3d = (P1-P0)[:3].reshape((3,)+shape).transpose((1,2,0))

    fid = int(flowl0s[index].split('/')[-1].split('_')[1])
    with open(calibs[index], 'r') as f:
        fid = fid - int(f.readline().split(' ')[-1])
    l_exts, r_exts= load_exts(calibs[index])
    if '/right/' in iml0s[index]:
        exts = r_exts
    else:
        exts = l_exts

    if '/into_future/' in flowl0s[index]:
        if (fid + 1) > len(exts) - 1: print(flowl0s[index])
        if (fid) > len(exts) - 1: print(flowl0s[index])
        ext1 = exts[fid+1]
        ext0 = exts[fid]
    else:
        if (fid - 1) > len(exts) - 1: print(flowl0s[index])
        if (fid) > len(exts) - 1: print(flowl0s[index])
        ext1 = exts[fid-1]
        ext0 = exts[fid]
    camT = np.eye(4); camT[1,1] = -1; camT[2,2] = -1  # Sceneflow uses Blender's coordinate system
    RT01 = camT.dot(np.linalg.inv(ext0)).dot(ext1).dot(camT)  # ext is from camera space to world space
    
    rect_flow3d = (RT01[:3,:3].dot(P1[:3])-P0[:3]).reshape((3,)+shape).transpose((1,2,0))  # rectified scene flow

    depthflow = np.concatenate((depth0[:,:,np.newaxis], rect_flow3d, flow3d), 2)
    RT01 = np.concatenate((cv2.Rodrigues(RT01[:3,:3])[0][:,0], RT01[:3,-1])).astype(np.float32)

    # object mask
    fnum = int(iml0s[index].split('/')[-1].split('.png')[0])
    obj_fname = '%s/%04d.pfm'%(flowl0s[index].replace('/optical_flow','object_index').replace('into_past/','/').replace('into_future/','/').rsplit('/',1)[0],fnum)
    obj_idx = disparity_loader(obj_fname)
    
    depthflow = np.concatenate((depthflow, obj_idx[:,:,np.newaxis]), 2)
    # depthflow dimension: H x W x 8 (depth=1 + rectified_flow3d=3 + flow3d=3 + object_segmentation=1)

    iml1 = np.asarray(iml1)
    iml0 = np.asarray(iml0)
    
    return iml0, iml1, flowl0, depthflow, intr, RT01


def motionmask(flowl0, depthflow, RT01):
    '''
    flowl0: optical flow. [H, W, 3]
    depthflow: a concatenation of depth, rectified scene flow, scene flow, and object segmentation. [H, W, 8]
    RT01: camera motion from the future frame to the current frame. [6, ]
    '''
    valid_mask = (flowl0[:,:,2] == 1) & (depthflow[:,:,0] < 100) & (depthflow[:,:,0] > 0.01)  # valid flow & valid depth
    Tglobal_gt = -RT01[3:, np.newaxis, np.newaxis]  # background translation
    Tlocal_gt = depthflow[:,:,1:4].transpose(2, 0, 1)   # point translation (after removing rotation)
    m3d_gt = np.linalg.norm(Tlocal_gt - Tglobal_gt, 2, 0)       # abs. motion
    fgmask_gt = m3d_gt * 100 > 1
    fgmask_gt[~valid_mask] = False

    return fgmask_gt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='segmask_gt_generation')
    parser.add_argument('--database',
                    help='path to the database (required)')
    parser.add_argument('--debug', action='store_true', default=False,
                    help='generate visualization')
    parser.add_argument('--frames_pass', default='frames_cleanpass', 
                    help='which pass to use, either clean or final')
    parser.add_argument('--dataset', 
                    help='choose from FlyingThings3D, Driving, Monkaa')
    args = parser.parse_args()

    if args.debug:
        os.makedirs('%s/%s/results_viz' % (args.database, args.dataset), exist_ok=True)
    
    if args.dataset == 'Monkaa':
        level = 4
    else:
        level = 6
    iml0s, iml1s, flowl0s, disp0s, dispcs, calibs = dataloader('%s/%s/' % (args.database, args.dataset), 
                                                                level=level, fpass=args.frames_pass)
    
    print("Generating %s masks..." % len(flowl0s))
    for i in range(len(iml0s)):
        idd = flowl0s[i].split('/')[-1].split('_')[-2]
        mask_fn = '%s/%s.npy' % (os.path.dirname(flowl0s[i]).replace('optical_flow', 'rigidmask'), idd)
        if os.path.exists(mask_fn):
            print(i)
            continue
        os.makedirs(os.path.dirname(mask_fn), exist_ok=True)

        iml0, iml1, flowl0, depthflow, intr, RT01 = exp_loader(i, iml0s, iml1s, flowl0s, disp0s, dispcs, calibs)
        fgmask = motionmask(flowl0, depthflow, RT01)
        np.save(mask_fn, fgmask)

        if args.debug:
            if args.dataset == 'Driving' and 'rigidmask/15mm_focallength/scene_forwards/fast/left' not in mask_fn:
                continue
            elif args.dataset == 'Monkaa' and 'rigidmask/eating_camera2_x2/left' not in mask_fn:
                continue
            elif args.dataset == 'FlyingThings3D' and not ('rigidmask/TEST/A' in mask_fn and 'into_future/left' in mask_fn):
                continue
            print("Visualizing %s" % mask_fn)
            flowl0viz = flow_to_image(flowl0)
            maskviz = np.stack((fgmask * 255.0, )*3, axis=-1).astype(np.uint8)
            inputs = np.concatenate([iml0, flowl0viz, maskviz], axis=1)
            cv2.imwrite('%s/%s/results_viz/%s.png' % (args.database, args.dataset, str(i).zfill(5)), cv2.cvtColor(inputs, cv2.COLOR_RGB2BGR))