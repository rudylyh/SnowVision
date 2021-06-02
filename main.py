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
parser.add_argument('--design-dir', type=str, default='./designs', help='a folder of design images')
parser.add_argument('--output', type=str, help='a zip file of matching results or a directory')
args = parser.parse_args()


opts = json.load(open(args.config_file, 'r'))
xyz_proc_lib = ctypes.CDLL("./libxyz_proc.so")
xyz_proc_lib.SplitCloud.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_double]


''' Load CNN models '''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda:1')
cen_net = seg_hrnet.get_seg_model('./model/hrnet_config.yaml', num_classes=2)
main_gpu = torch.device('cuda:1')
cen_net = torch.nn.DataParallel(cen_net.to(main_gpu), device_ids=[1,0])
checkpoint = torch.load('./model/hrnet_seg.pth')
cen_net.module.load_state_dict(checkpoint['state_dict'])
cmn_net = CMN()
cmn_ckp = torch.load('./model/cmn.pth', map_location=lambda storage, loc: storage)
cmn_net.load_state_dict(cmn_ckp)
cmn_net.to(device)
matcher = Matcher(opts, cmn_net, args.design_dir)


if __name__ == "__main__":
    if args.input.endswith('zip'):
        # Create two temporary folders to store inputs and outputs
        in_dir = './tmp_input'
        out_dir = './tmp_output'
        os.mkdir(in_dir)
        os.mkdir(out_dir)
        # Unzip all files and process one by one
        zip_reader = ZipFile(args.input, 'r')
        zip_reader.extractall(in_dir)
        zip_reader.close()
    else:
        in_dir = args.input
        out_dir = args.output

    if args.scan_name is None:
        name_list = os.listdir(args.in_dir)
    else:
        name_list = [args.scan_name]

    for i, scan_name in enumerate(name_list):
        print("Splitting scan:", scan_name, f'({i}/{len(name_list)})')
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
            sherd_name = scan_name.split('.')[0] + '-' + str(i+1)
            print("--- Processing sherd:", sherd_name)
            try:
                pt_cloud = ReadXYZ(os.path.join(out_dir, sherd_name, 'sherd.xyz'))

                print("------ Extracting depth image")
                depth_mat, depth_img, mask_img = PointCloud2DepthImg(pt_cloud, px_size=opts['sample_resolution'])
                cv2.imwrite(os.path.join(out_dir, sherd_name, 'depth.png'), depth_img)
                cv2.imwrite(os.path.join(out_dir, sherd_name, 'mask.png'), mask_img)

                print("------ Extracting curve image")
                curve_img = depth2curve(depth_img, mask_img, cen_net)
                cv2.imwrite(os.path.join(out_dir, sherd_name, 'curve.png'), curve_img)

                print("------ Matching with all designs")
                top_k_match = matcher.GetTopKMatch(sherd_name.split('.')[0], depth_img, curve_img, mask_img, args.design_dir)
                matcher.WriteMatchResults(top_k_match, os.path.join(out_dir, sherd_name))
            except:
                print("------ Failed.")

    if args.output.endswith('zip'):
        shutil.make_archive(args.output[:-4], 'zip', out_dir)
        shutil.rmtree(in_dir)
        shutil.rmtree(out_dir)
