import cv2
import torch
import numpy as np
from skimage.morphology import skeletonize


def Normalize(image):
    min_val = min(map(min, image))
    image = image - min_val
    max_val = max(map(max, image))
    image = image / max_val
    image = image * 255
    return image.astype(np.uint8)


def ImagePadding(img, k):
    padded_img = np.zeros((img.shape[0] + 2*k, img.shape[1] + 2*k), np.uint8)
    padded_img[k:k+img.shape[0],k:k+img.shape[1]] = img
    return padded_img


def Softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def RemoveSmallArea(img, thre):
    # img2, contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < thre:
            cv2.fillPoly(img, pts =[contour], color=(0))


#depth image to skeleton heatmap
def depth2hm(depth_img, cen_net):
    cen_input = depth_img[np.newaxis, np.newaxis, ...]
    cen_input = torch.Tensor(cen_input).to(torch.device('cuda'))
    hm_img = cen_net(cen_input).data.cpu().numpy()
    hm_img = Normalize(hm_img[0,1])
    # cv2.imshow('depth_img', depth_img)
    # cv2.imshow('hm_img', hm_img)
    # cv2.waitKey()
    return hm_img


#skeleton heatmap to binary skeleton image
def hm2skel(hm_img):
    raw_curve_img = np.zeros((hm_img.shape), np.uint8)
    raw_curve_img[hm_img<=128] = 0;
    raw_curve_img[hm_img>128] = 1;
    skel_bool_img = skeletonize(raw_curve_img)
    raw_skel_img = np.zeros((raw_curve_img.shape), np.uint8)
    raw_skel_img[skel_bool_img == True] = 255
    return raw_skel_img


#refine raw skeleton image, the size of cropped image patch is 2k+1
def RefineSkel(hm_img, raw_skel_img, pcn_net, k = 25):
    pad_hm_img = ImagePadding(hm_img, k)
    pad_raw_skel_img = ImagePadding(raw_skel_img, k)
    skel_img = np.zeros(raw_skel_img.shape, np.uint8)
    for r in range(k, pad_raw_skel_img.shape[0]-k):
        for c in range(k, pad_raw_skel_img.shape[1]-k):
            if pad_raw_skel_img[r,c] == 255:
                hm_patch = pad_hm_img[r-k:r+k+1, c-k:c+k+1]
                hm_patch = cv2.cvtColor(hm_patch, cv2.COLOR_GRAY2RGB)
                hm_patch = np.transpose(hm_patch, (2,0,1))
                hm_patch = hm_patch[np.newaxis, ...]
                hm_patch = torch.Tensor(hm_patch).to(torch.device('cuda'))
                prob = Softmax(pcn_net(hm_patch).data.cpu().numpy()[0])[1]
                if prob > 0.5:
                    skel_img[r-k, c-k] = 255
    return skel_img


#recover skeleton image to curve image
def skel2curve(depth_img, mask_img, skel_img):
    ori_curve_img = np.zeros(depth_img.shape, np.uint8)
    p = 9 #search radius
    for r in range(skel_img.shape[0]):
        for c in range(skel_img.shape[1]):
            if skel_img[r,c]==255:
                (min_val, max_val) = (255, 0)
                pixel_list = []
                for x in range(max(0, r-p), min(skel_img.shape[0], r+p+1)):
                    for y in range(max(0, c-p), min(skel_img.shape[1], c+p+1)):
                        if (r-x)*(r-x)+(c-y)*(c-y) < p*p:
                            pixel_list.append((x,y))
                            if depth_img[x,y] > max_val:
                                max_val = depth_img[x,y]
                            if depth_img[x,y] < min_val:
                                min_val = depth_img[x,y]
                thre = min_val + (depth_img[r,c]-min_val)/2.0
                for pixel in pixel_list:
                    if depth_img[pixel] > thre:
                        ori_curve_img[pixel] = 255
    curve_img = cv2.GaussianBlur(ori_curve_img, (27,27), 0);
    ret, curve_img = cv2.threshold(curve_img, 127, 255, cv2.THRESH_BINARY)
    RemoveSmallArea(curve_img, 500)
    curve_img[mask_img == 0] = 0
    return curve_img


def depth2curve(depth_img, mask_img, cen_net, pcn_net):
    hm_img = depth2hm(depth_img, cen_net)
    raw_skel_img = hm2skel(hm_img)
    skel_img = RefineSkel(hm_img, raw_skel_img, pcn_net)
    curve_img = skel2curve(depth_img, mask_img, skel_img)
    return curve_img
