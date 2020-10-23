import os
import ctypes
import numpy as np
import cv2
import argparse
import json
from collections import OrderedDict
from curve_extraction import depth2curve
from curve_matching import Matcher
from xyz2depth import ReadXYZ, PointCloud2DepthImg
import torch
from network import CEN, CMN, PCN
import time
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--config-file', type=str, default='./config.json', help='json of parameter settings')
parser.add_argument('--in-dir', type=str, default='/WD1/SnowVision/data/Karen201012/orig_xyz', help='a folder of raw scans')
parser.add_argument('--scan-name', type=str, default=None, help='process the whole input folder if None')
parser.add_argument('--design-dir', type=str, default='/WD1/SnowVision/data/designs', help='a folder of design images')
parser.add_argument('--out-dir', type=str, default='/WD1/SnowVision/data/Karen201012/matching_result', help='one sub-folders for each sherd')
args = parser.parse_args()


opts = json.load(open(args.config_file, 'r'))
xyz_proc_lib = ctypes.CDLL("./libxyz_proc.so")
xyz_proc_lib.SplitCloud.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_double]


''' Load CNN models '''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda:1')
cen_net = CEN()
cen_ckp = torch.load('./model/cen.pth', map_location=lambda storage, loc: storage)
cen_net.load_state_dict(cen_ckp)
cen_net.to(device)
pcn_net = PCN()
pcn_ckp = torch.load('./model/pcn.pth', map_location=lambda storage, loc: storage)
pcn_net.load_state_dict(pcn_ckp)
pcn_net.to(device)
cmn_net = CMN()
cmn_ckp = torch.load('./model/cmn.pth', map_location=lambda storage, loc: storage)
cmn_net.load_state_dict(cmn_ckp)
cmn_net.to(device)
matcher = Matcher(opts, cmn_net, args.design_dir)

# sherd_name_list = ['SCAN001374-3']
sherd_name_list = ['SCAN001350-5', 'SCAN001351-4', 'SCAN001357-4', 'SCAN001360-1', 'SCAN001362-2', 'SCAN001363-3', 'SCAN001364-2', 'SCAN001365-1', 'SCAN001366-1', 'SCAN001367-2', 'SCAN001367-3', 'SCAN001369-1', 'SCAN001369-2', 'SCAN001370-1', 'SCAN001370-2', 'SCAN001371-1', 'SCAN001374-3', 'SCAN001378-1', 'SCAN001379-1', 'SCAN001381-1', 'SCAN001383-1', 'SCAN001386-2', 'SCAN001386-3']
scan_name_list = [sherd_name.split('-')[0]+'.xyz' for sherd_name in sherd_name_list]
for sherd_name in os.listdir('/WD1/SnowVision/data/Karen201012/matching_result'):
    if sherd_name in sherd_name_list:
        shutil.copyfile('/WD1/SnowVision/data/Karen201012/matching_result/'+sherd_name+'/match_result.xlsx', '/WD1/SnowVision/data/Karen201012/hartford_matching_result_1021/'+sherd_name+'.xlsx')

if __name__ == "__main__":
    if args.scan_name is None:
        name_list = os.listdir(args.in_dir)
    else:
        name_list = [args.scan_name]

    for scan_name in name_list:
        # if os.path.exists(os.path.join(args.out_dir, scan_name.split('.')[0]+'-1')):
        #     continue
        if scan_name not in scan_name_list:
            continue
        print("Splitting scan:", scan_name)
        split_num = xyz_proc_lib.SplitCloud(
                        bytes(args.in_dir, encoding='utf8'),
                        bytes(scan_name, encoding='utf8'),
                        bytes(args.out_dir, encoding='utf8'),
                        opts['min_pc_height'],
                        opts['min_size_percent'])
        if split_num == -1:
            print("No sherd found.")
        else:
            print(split_num, "sherd(s) found.")

        for i in range(0, split_num):
            sherd_name = scan_name.split('.')[0] + '-' + str(i+1)
            if sherd_name not in sherd_name_list:
                continue
            print("--- Processing sherd:", sherd_name)
            pt_cloud = ReadXYZ(os.path.join(args.out_dir, sherd_name, 'sherd.xyz'))

            print("------ Extracting depth image")
            depth_mat, depth_img, mask_img = PointCloud2DepthImg(pt_cloud, px_size=opts['sample_resolution'])
            cv2.imwrite(os.path.join(args.out_dir, sherd_name, 'depth.png'), depth_img)
            cv2.imwrite(os.path.join(args.out_dir, sherd_name, 'mask.png'), mask_img)

            print("------ Extracting curve image")
            curve_img = depth2curve(depth_img, mask_img, cen_net, pcn_net)
            cv2.imwrite(os.path.join(args.out_dir, sherd_name, 'curve.png'), curve_img)

            print("------ Matching with all designs")

            # depth_img = cv2.imread('/WD1/SnowVision/exp_tip/val_data/depth/1002_86.png', 0)
            # curve_img = cv2.imread('/WD1/SnowVision/exp_tip/val_data/curve/1002_86.png', 0)
            # mask_img = cv2.imread('/WD1/SnowVision/exp_tip/val_data/mask/1002_86.png', 0)
            # design_dir = '/WD1/SnowVision/exp_tip/val_data/design'
            # t1 = time.time()
            top_k_match = matcher.GetTopKMatch(sherd_name.split('.')[0], depth_img, curve_img, mask_img, args.design_dir)
            # t2 = time.time()
            # print(t2-t1)
            # matcher.WriteMatchResults(top_k_match, os.path.join(args.out_dir, sherd_name))
