import os
import ctypes
import numpy as np
import cv2
import caffe
import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument('--config-file', type=str, default='./config.json', help='parameter settings')
parser.add_argument('--src-dir', type=str, default='./data/raw_xyz/')
parser.add_argument('--xyz-name', type=str)
parser.add_argument('--dst-dir', type=str, default='./data/split_xyz/')
args = parser.parse_args()


opts = json.load(open(args.config_file, 'r'))
xyz_proc_lib = ctypes.CDLL("libxyz_proc.so")
xyz_proc_lib.SplitCloud.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_double]


if __name__ == "__main__":
    if args.xyz_name is None:
        name_list = os.listdir(args.src_dir)
    else:
        name_list = [args.xyz_name]

    for xyz_name in name_list:
        print("Splitting sherd:", xyz_name)
        split_num = xyz_proc_lib.SplitCloud(
                        bytes(args.src_dir, encoding='utf8'),
                        bytes(xyz_name, encoding='utf8'),
                        bytes(args.dst_dir, encoding='utf8'),
                        opts['min_height_percent'],
                        opts['min_size_percent'])
        if split_num == -1:
            print("Failed.")
        else:
            print(split_num, "sherd(s) split.")
