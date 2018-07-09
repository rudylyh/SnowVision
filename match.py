import os
os.environ['GLOG_minloglevel'] = '2'
import ctypes
import numpy
import cv2
import caffe
import depth2curve
import curve2design

caffe.set_mode_cpu()
#caffe.set_mode_gpu()
#caffe.set_device(0)
xyz_dir = "./data/xyz/"
split_xyz_dir = "./data/split_xyz/"
depth_dir = "./data/depth/"
mask_dir = "./data/mask/"
curve_dir = "./data/curve/"
design_dir = './data/design/'
cropped_patch_dir = "./data/cropped_patch/"
marked_design_dir = "./data/marked_design/"

myLib = ctypes.CDLL("./sherd2depth.so")
myLib.SplitCloud.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_double]
myLib.xyz2depth.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double]

cmn_net = caffe.Net('./model/cmn_deploy.prototxt','./model/cmn_iter_10000.caffemodel', caffe.TEST)
cen_net = caffe.Net('./model/cen_deploy.prototxt','./model/cen_iter_27000.caffemodel', caffe.TEST)
pcn_net = caffe.Net('./model/pcn_deploy.prototxt','./model/pcn_iter_50000.caffemodel', caffe.TEST)

min_height_percent = 0.2
min_size_percent = 0.01
sample_resolution = 0.1
resize_scale = 0.5
return_top_k = 3

def split(xyz_name):
    print "Splitting", xyz_name, ":"
    split_num = myLib.SplitCloud(xyz_dir + file_name, split_xyz_dir, min_height_percent, min_size_percent)
    if split_num == -1:
        print "Cannot split this file."
    else:
        print split_num, "sherd(s) found."

def match(xyz_name):
    print "Processing", xyz_name, ":"
    img_name = xyz_name.split('.')[0] + '.png'
    print "Extracting depth image ..."
    proc_status = myLib.xyz2depth(split_xyz_dir + xyz_name, depth_dir + img_name, mask_dir + img_name, sample_resolution)
    if proc_status == -1:
        print "Failed to extract depth image."
        return
    print "Extracting curve image ..."
    depth_img = cv2.imread(depth_dir + img_name, 0)
    mask_img = cv2.imread(mask_dir + img_name, 0)
    curve_img = depth2curve.main(depth_img, mask_img, cen_net, pcn_net)
    cv2.imwrite(curve_dir + img_name, curve_img)
    print "Matching with the design dataset ..."
    match_result = curve2design.main(curve_img, mask_img, design_dir, cmn_net, resize_scale, return_top_k)
    print "The best matched design is:", match_result[0][0].split('.')[0]
    cv2.imwrite(cropped_patch_dir + img_name, match_result[0][-2])
    cv2.imwrite(marked_design_dir + img_name, match_result[0][-1])
    print "Finished. \n"

if __name__ == "__main__":
    '''
    # Split the input file if it has multiple sherds
    for file_name in os.listdir(xyz_dir):
        split(file_name)
    '''
    for file_name in os.listdir(split_xyz_dir):
        #if file_name != "164-84-3-3.xyz":
        #    continue
        match(file_name)
