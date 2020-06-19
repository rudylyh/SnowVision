import os
import ctypes
import numpy as np
import cv2
import caffe
import argparse
import json
from curve_extraction import depth2curve
from curve_matching import Matcher


parser = argparse.ArgumentParser()
parser.add_argument('--config-file', type=str, default='./config.json', help='parameter settings')
parser.add_argument('--xyz-dir', type=str, default='./data/split_xyz/')
parser.add_argument('--xyz-name', type=str, default=None)
parser.add_argument('--design-dir', type=str, default='./data/designs/', help='design database')
parser.add_argument('--depth-dir', type=str, default='./data/depth/', help='for saving depth images')
parser.add_argument('--curve-dir', type=str, default='./data/curve/', help='for saving extracted curves')
parser.add_argument('--mask-dir', type=str, default='./data/mask/', help='for saving mask images')
parser.add_argument('--res-dir', type=str, default='./data/match_result/')
args = parser.parse_args()


opts = json.load(open(args.config_file, 'r'))
xyz_proc_lib = ctypes.CDLL("libxyz_proc.so")
xyz_proc_lib.xyz2depth.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double]


''' Load CNN models '''
# caffe.set_mode_cpu()
caffe.set_mode_gpu()
caffe.set_device(0)
cmn_net = caffe.Net('./model/cmn_deploy.prototxt','./model/cmn_iter_10000.caffemodel', caffe.TEST)
cen_net = caffe.Net('./model/cen_deploy.prototxt','./model/cen_iter_27000.caffemodel', caffe.TEST)
pcn_net = caffe.Net('./model/pcn_deploy.prototxt','./model/pcn_iter_50000.caffemodel', caffe.TEST)
matcher = Matcher(opts, cmn_net, args.design_dir)


if __name__ == "__main__":
    if args.xyz_name is None:
        name_list = os.listdir(args.xyz_dir)
    else:
        name_list = [args.xyz_name]

    for xyz_name in name_list:
        print("Processing:", xyz_name)
        img_name = xyz_name.split('.')[0] + '.png'

        print("Extracting depth image ...")
        proc_status = xyz_proc_lib.xyz2depth(
                        bytes(args.xyz_dir + xyz_name, encoding='utf8'),
                        bytes(args.depth_dir + img_name, encoding='utf8'),
                        bytes(args.mask_dir + img_name, encoding='utf8'),
                        opts['sample_resolution'])
        if proc_status == -1:
            print("Failed to extract depth image.")
            continue

        print("Extracting curve image ...")
        depth_img = cv2.imread(args.depth_dir + img_name, 0)
        mask_img = cv2.imread(args.mask_dir + img_name, 0)
        curve_img = depth2curve(depth_img, mask_img, cen_net, pcn_net)
        cv2.imwrite(args.curve_dir + img_name, curve_img)

        print("Matching with designs ...")
        top_k_match = matcher.GetTopKMatch(xyz_name.split('.')[0], depth_img, curve_img, mask_img, args.design_dir)
        matcher.WriteMatchResults(top_k_match, args.res_dir)

        print("Finished.")
