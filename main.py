import os
import ctypes
import numpy as np
import cv2
import caffe
import argparse
import json
import torch
from torch import nn
from collections import OrderedDict
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
xyz_proc_lib = ctypes.CDLL("/WD1/github_repos/SnowVision/libxyz_proc.so")
xyz_proc_lib.xyz2depth.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double]


class CurveNet(nn.Module):
    def __init__(self):
        super(CurveNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),  # section 3.3
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(96, 256, 5, padding=2, groups=2),  # (b x 256 x 27 x 27)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1, groups=2),  # (b x 384 x 13 x 13)
            # nn.ReLU(),
            # nn.AvgPool2d(kernel_size=13, stride=1),  # (b x 256 x 6 x 6)
            )

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.shape[0], -1)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def LoadCurveNetModel(ckp_path):
    ori_ckp = torch.load(ckp_path, map_location=lambda storage, loc: storage)
    renamed_ckp = OrderedDict()
    renamed_ckp['net.0.weight'] = ori_ckp['conv1.weight']
    renamed_ckp['net.0.bias'] = ori_ckp['conv1.bias']
    renamed_ckp['net.4.weight'] = ori_ckp['conv2.weight']
    renamed_ckp['net.4.bias'] = ori_ckp['conv2.bias']
    renamed_ckp['net.8.weight'] = ori_ckp['conv3.weight']
    renamed_ckp['net.8.bias'] = ori_ckp['conv3.bias']
    renamed_ckp['net.10.weight'] = ori_ckp['conv4.weight']
    renamed_ckp['net.10.bias'] = ori_ckp['conv4.bias']
    model = CurveNet()
    model.load_state_dict(renamed_ckp)
    return model.to(device)
# sim_model = LoadCurveNetModel('./model/sim_model.pth')
# matcher = Matcher(opts, sim_model, args.design_dir)


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
