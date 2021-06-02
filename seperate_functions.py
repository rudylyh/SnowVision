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
from zipfile import ZipFile
import shutil
import seg_hrnet


parser = argparse.ArgumentParser()
parser.add_argument('--config-file', type=str, default='./config.json', help='json of parameter settings')
parser.add_argument('--input', type=str, help='a zip file of raw scans or a directory')
parser.add_argument('--scan-name', type=str, default=None, help='process the whole input folder if None')
parser.add_argument('--design-dir', type=str, default='/WD1/SnowVision/data/designs', help='a folder of design images')
parser.add_argument('--output', type=str, help='a zip file of matching results or a directory')
args = parser.parse_args()


opts = json.load(open(args.config_file, 'r'))
xyz_proc_lib = ctypes.CDLL("./libxyz_proc.so")
xyz_proc_lib.SplitCloud.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_double]


''' Load CNN models '''
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda:1')
cen_net = seg_hrnet.get_seg_model('./model/hrnet_config.yaml', num_classes=2)
main_gpu = torch.device('cuda:1')
cen_net = torch.nn.DataParallel(cen_net.to(main_gpu), device_ids=[1,0])
checkpoint = torch.load('./model/hrnet_seg.pth')
cen_net.module.load_state_dict(checkpoint['state_dict'])
cmn_net = CMN()
cmn_ckp = torch.load('./model/cmn.pth', map_location=lambda storage, loc: storage)
cmn_net.load_state_dict(cmn_ckp)
cmn_net.to(main_gpu)
matcher = Matcher(opts, cmn_net, args.design_dir)


def get_depth():
    in_dir = '/WD1/SnowVision/data/usf_xyx_batch1/original_xyz'
    out_dir = '/WD1/SnowVision/data/usf_xyx_batch1/output'
    for i, scan_name in enumerate(os.listdir(in_dir)):
        print(i, scan_name)
        split_num = xyz_proc_lib.SplitCloud(
                        bytes(in_dir, encoding='utf8'),
                        bytes(scan_name, encoding='utf8'),
                        bytes(out_dir, encoding='utf8'),
                        opts['min_pc_height'],
                        opts['min_size_percent'])
        if split_num == -1:
            print("No sherd found.")
        else:
            print(split_num, "sherd(s) found.")

        for i in range(0, split_num):
            sherd_name = os.path.splitext(scan_name)[0] + '-' + str(i+1)
            print("--- Processing sherd:", sherd_name)
            # try:
            pt_cloud = ReadXYZ(os.path.join(out_dir, sherd_name, 'sherd.xyz'))

            print("------ Extracting depth image")
            depth_mat, depth_img, mask_img = PointCloud2DepthImg(pt_cloud, px_size=opts['sample_resolution'])
            cv2.imwrite(os.path.join(out_dir, sherd_name, 'depth.png'), depth_img)
            cv2.imwrite(os.path.join(out_dir, sherd_name, 'mask.png'), mask_img)


def match_curve():
    data_dir = '/home/rudy/WD1/SnowVision/exp_lsc_scan'
    save_dir = '/WD1/SnowVision/data/Karen20210111_top100/top100_match_result'
    sherd_list = os.listdir('/WD1/SnowVision/data/Karen20210111_top100/top100_sherds')
    for i, sherd_name in enumerate(sherd_list):
        sherd_name = sherd_name.split('.')[0]
        print(i, sherd_name)
        depth_img = cv2.imread(os.path.join(data_dir, 'depth', sherd_name+'.png'), 0)
        mask_img = cv2.imread(os.path.join(data_dir, 'mask', sherd_name+'.png'), 0)
        # curve_img = cv2.imread(os.path.join(src_dir, sherd_name, 'curve.png'), 0)
        curve_img = depth2curve(depth_img, mask_img, cen_net)
        # cv2.imshow('depth_img', depth_img)
        # cv2.imshow('curve_img', curve_img)
        # cv2.waitKey()
        top_k_match = matcher.GetTopKMatch(sherd_name, depth_img, curve_img, mask_img, args.design_dir)
        tmp_save_dir = os.path.join(save_dir, sherd_name)
        if not os.path.exists(tmp_save_dir):
            os.mkdir(tmp_save_dir)
        shutil.copyfile(os.path.join(data_dir, 'depth', sherd_name+'.png'), os.path.join(tmp_save_dir, 'depth.png'))
        shutil.copyfile(os.path.join(data_dir, 'mask', sherd_name+'.png'), os.path.join(tmp_save_dir, 'mask.png'))
        cv2.imwrite(os.path.join(tmp_save_dir, 'curve.png'), curve_img)
        matcher.WriteMatchResults(top_k_match, tmp_save_dir)


if __name__ == "__main__":
    get_depth()
