from torch.utils.data import DataLoader
from Datasets.utils import ToTensor, Compose, CropCenter, ResizeData, dataset_intrinsics, DownscaleFlow, plot_traj, visflow, load_kiiti_intrinsics
from Datasets.tartanTrajFlowDataset import TrajFolderDataset
from evaluator.transformation import pose_quats2motion_ses, motion_ses2pose_quats
from evaluator.tartanair_evaluator import TartanAirEvaluator
from evaluator.evaluator_base import per_frame_scale_alignment
from DytanVO import DytanVO

import argparse
import numpy as np
import cv2
from os import mkdir
from os.path import isdir

def get_args():
    parser = argparse.ArgumentParser(description='Inference code of DytanVO')

    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size (default: 1)')
    parser.add_argument('--worker-num', type=int, default=1,
                        help='data loader worker number (default: 1)')
    parser.add_argument('--image-width', type=int, default=640,
                        help='image width (default: 640)')
    parser.add_argument('--image-height', type=int, default=448,
                        help='image height (default: 448)')
    parser.add_argument('--vo-model-name', default='',
                        help='name of pretrained VO model (default: "")')
    parser.add_argument('--flow-model-name', default='',
                        help='name of pretrained flow model (default: "")')
    parser.add_argument('--pose-model-name', default='',
                        help='name of pretrained pose model (default: "")')
    parser.add_argument('--seg-model-name', default='',
                        help='name of pretrained segmentation model (default: "")')
    parser.add_argument('--airdos', action='store_true', default=False,
                        help='airdos test (default: False)')
    parser.add_argument('--rs_d435', action='store_true', default=False,
                        help='realsense d435i test (default: False)')
    parser.add_argument('--sceneflow', action='store_true', default=False,
                        help='sceneflow test (default: False)')
    parser.add_argument('--kitti', action='store_true', default=False,
                        help='kitti test (default: False)')
    parser.add_argument('--kitti-intrinsics-file',  default='',
                        help='kitti intrinsics file calib.txt (default: )')
    parser.add_argument('--test-dir', default='',
                        help='test trajectory folder where the RGB images are (default: "")')
    parser.add_argument('--pose-file', default='',
                        help='test trajectory gt pose file, used for scale calculation, and visualization (default: "")')
    parser.add_argument('--save-flow', action='store_true', default=False,
                        help='save optical flow (default: False)')
    parser.add_argument('--seg-thresh', type=float, default=0.7,
                        help='threshold for motion segmentation')
    parser.add_argument('--iter-num', type=int, default=2,
                        help='number of iterations')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    testvo = DytanVO(args.vo_model_name, args.seg_model_name, args.image_height, args.image_width, 
                    args.kitti, args.flow_model_name, args.pose_model_name)

    # load trajectory data from a folder
    if args.kitti:
        datastr = 'kitti'
    elif args.airdos:
        datastr = 'airdos'
    elif args.rs_d435:
        datastr = 'rs_d435'
    elif args.sceneflow:
        datastr = 'sceneflow'
    else:
        datastr = 'tartanair'
    focalx, focaly, centerx, centery, baseline = dataset_intrinsics(datastr) 
    if args.kitti_intrinsics_file.endswith('.txt') and datastr == 'kitti':
        focalx, focaly, centerx, centery, baseline = load_kiiti_intrinsics(args.kitti_intrinsics_file)

    if datastr == 'kitti':
        transform = Compose([ResizeData((args.image_height, 1226)), CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])
    else:
        transform = Compose([CropCenter((args.image_height, args.image_width)), DownscaleFlow(), ToTensor()])

    testDataset = TrajFolderDataset(args.test_dir, transform=transform, 
                                        focalx=focalx, focaly=focaly, centerx=centerx, centery=centery)
    testDataloader = DataLoader(testDataset, batch_size=args.batch_size, 
                                        shuffle=False, num_workers=args.worker_num)
    testDataiter = iter(testDataloader)

    motionlist = []
    testname = datastr + '_' + args.vo_model_name.split('.')[0]
    if args.save_flow:
        flowdir = 'results/'+testname+'_flow'
        if not isdir(flowdir):
            mkdir(flowdir)
        flowcount = 0
    while True:
        try:
            sample = testDataiter.next()
        except StopIteration:
            break

        motion, flow = testvo.test_batch(sample, [focalx, centerx, centery, baseline], args.seg_thresh, args.iter_num)
        motionlist.append(motion)

        if args.save_flow:
            for k in range(flow.shape[0]):
                flowk = flow[k].transpose(1,2,0)
                np.save(flowdir+'/'+str(flowcount).zfill(6)+'.npy',flowk)
                flow_vis = visflow(flowk)
                cv2.imwrite(flowdir+'/'+str(flowcount).zfill(6)+'.png',flow_vis)
                flowcount += 1

    motions = np.array(motionlist)

    # calculate ATE, RPE, KITTI-RPE
    if args.pose_file.endswith('.txt'):
        gtposes = np.loadtxt(args.pose_file)
        if datastr == 'airdos':
            gtposes = gtposes[:,1:]  # remove the first column of timestamps
        gtmotions = pose_quats2motion_ses(gtposes)
        estmotion_scale = per_frame_scale_alignment(gtmotions, motions)
        estposes = motion_ses2pose_quats(estmotion_scale)

        evaluator = TartanAirEvaluator()
        results = evaluator.evaluate_one_trajectory(gtposes, estposes, scale=True, kittitype=(datastr=='kitti'))
        
        print("==> ATE: %.4f,\t KITTI-R/t: %.4f, %.4f" %(results['ate_score'], results['kitti_score'][0], results['kitti_score'][1]))

        # save results and visualization
        plot_traj(results['gt_aligned'], results['est_aligned'], vis=False, savefigname='results/'+testname+'.png', title='ATE %.4f' %(results['ate_score']))
        np.savetxt('results/'+testname+'.txt',results['est_aligned'])
    else:
        np.savetxt('results/'+testname+'.txt', motion_ses2pose_quats(motions))