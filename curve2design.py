import os
import sys
import cv2
import math
import numpy
import random
import time

def RotateImg(image, angle, scale = 1.0):
    (h, w) = image.shape[:2]
    diag = int(math.ceil(math.sqrt(h*h+w*w)))
    diag_img = numpy.zeros([diag, diag], dtype=numpy.uint8)
    diag_img[(diag-h)/2:(diag+h)/2, (diag-w)/2:(diag+w)/2] = image
    M = cv2.getRotationMatrix2D((diag/2, diag/2), angle, scale)
    rotated_img = cv2.warpAffine(diag_img, M, (diag, diag))
    return rotated_img

def GetTMResult(curve_img, mask_img, design_img):
    min_cost = sys.maxint
    for angle in range(0, 360):
        rotated_curve = RotateImg(curve_img, angle)
        rotated_mask = RotateImg(mask_img, angle)
        heatmap = cv2.matchTemplate(design_img, rotated_curve, cv2.TM_SQDIFF, None, rotated_mask)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap)
        if min_val < min_cost:
            tm_match_pos = [angle, min_loc[1], min_loc[0]]
            min_cost = min_val
    return tm_match_pos

# crop an image patch from a design image by the matching position
def GetDesignPatch(match_pos, mask_img, design_img):
    angle = float(match_pos[0])
    dx = int(match_pos[1])
    dy = int(match_pos[2])
    rotated_mask = RotateImg(mask_img, angle)
    patch = design_img[dx:dx+rotated_mask.shape[0], dy:dy+rotated_mask.shape[1]].copy()
    patch[rotated_mask==0] = 0
    diag = rotated_mask.shape[0]
    M = cv2.getRotationMatrix2D((diag/2, diag/2), -angle, 1.0)
    rotated_patch = cv2.warpAffine(patch, M, (diag, diag))
    cropped_patch = rotated_patch[(rotated_patch.shape[0]-mask_img.shape[0])/2:(rotated_patch.shape[0]+mask_img.shape[0])/2, (rotated_patch.shape[1]-mask_img.shape[1])/2:(rotated_patch.shape[1]+mask_img.shape[1])/2]
    cropped_patch[mask_img==0] = 0
    return cropped_patch

# mark sherd's location on the design image
def MarkDesign(match_pos, mask_img, design_img):
    angle = float(match_pos[0])
    dx = int(match_pos[1])
    dy = int(match_pos[2])
    rotated_mask = RotateImg(mask_img, angle)
    design_mask = numpy.zeros(design_img.shape, dtype = numpy.uint8)
    for i in range(0, rotated_mask.shape[0]):
        for j in range(0, rotated_mask.shape[1]):
            if rotated_mask[i][j]  == 255:
                design_mask[dx + i][dy + j] = 255
    im, contours, hierarchy = cv2.findContours(design_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    design_img = cv2.cvtColor(design_img, cv2.COLOR_GRAY2RGB)
    marked_design = cv2.drawContours(design_img, contours, -1, (0,255,0), 3)
    return marked_design

def img2feat(img, cmn_net, img_size = 227):
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(float)
    img = img - 128
    img = img / 255
    img = img[numpy.newaxis, ...]
    cmn_net.blobs['data'].reshape(1, *img.shape)
    cmn_net.blobs['data'].data[...] = img
    cmn_net.forward()
    feat = cmn_net.blobs['feat'].data.copy()
    return feat

def main(curve_img, mask_img, design_dir, cmn_net, resize_scale, k):
    curve_img = cv2.resize(curve_img, None, fx = resize_scale, fy = resize_scale, interpolation = cv2.INTER_CUBIC)
    mask_img = cv2.resize(mask_img, None, fx = resize_scale, fy = resize_scale, interpolation = cv2.INTER_CUBIC)
    match_list = []
    (h, w) = curve_img.shape[:2]
    diag = int(math.ceil(math.sqrt(h*h+w*w)))
    sherd_feat = img2feat(curve_img, cmn_net)
    for design_name in sorted(os.listdir(design_dir)):
        #if design_name != "Design48.png":
        #    continue
        design_img = cv2.imread(design_dir + design_name, 0)
        design_img = cv2.resize(design_img, None, fx = resize_scale, fy = resize_scale, interpolation = cv2.INTER_CUBIC)
        if diag <= design_img.shape[0] and diag <= design_img.shape[1]:
            tm_match_pos = GetTMResult(curve_img, mask_img, design_img)
            temp_patch = GetDesignPatch(tm_match_pos, mask_img, design_img)
            design_feat = img2feat(temp_patch, cmn_net)
            match_cost = numpy.linalg.norm(sherd_feat - design_feat)
            if len(match_list) < k or match_cost < match_list[k-1][4]:
                marked_design = MarkDesign(tm_match_pos, mask_img, design_img)
                temp_match = [design_name, tm_match_pos[0], tm_match_pos[1], tm_match_pos[2], match_cost, temp_patch, marked_design]
                match_list.append(temp_match)
                match_list.sort(key=lambda x:x[4])
                if len(match_list) > k:
                    del match_list[k:]
    return match_list
