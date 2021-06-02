import cv2
import torch
import numpy as np
import torchvision.transforms.functional as TF


def RemoveSmallArea(img, thre):
    # img2, contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < thre:
            cv2.fillPoly(img, pts =[contour], color=(0))
    return img


def depth2curve(depth_img, mask_img, cen_net):
    norm_depth_img = TF.to_tensor(depth_img)
    norm_depth_img = TF.normalize(norm_depth_img, mean=0.5032, std=0.3554)
    input = norm_depth_img.repeat(3,1,1).unsqueeze(0)
    output = cen_net(input)
    pred = output.data.cpu().numpy()
    pred = np.argmax(pred, axis=1)[0]
    pred = RemoveSmallArea(pred.astype(np.uint8), 100)
    curve_img = (255*pred).astype(np.uint8)
    return curve_img
