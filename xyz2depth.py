import os
import cv2
import numpy as np
import faiss


def ReadXYZ(filename):
   coordinates = []
   xyz = open(filename)
   title = xyz.readline()
   for line in xyz:
       x,y,z = line.split()
       coordinates.append([float(x), float(y), float(z)])
   xyz.close()
   return np.asarray(coordinates)


def PointCloud2DepthImg(pt_array, px_size = 0.1):
    pt_array[:,0] -= np.min(pt_array[:,0])
    pt_array[:,1] -= np.min(pt_array[:,1])
    pt_array[:,2] -= np.min(pt_array[:,2])
    max_x, max_y = np.max(pt_array[:,0]), np.max(pt_array[:,1])

    # Get N*2 coordinates of points and H*W*2 coordinates of pixels
    px_x, px_y = np.meshgrid(np.arange(0, max_x+px_size, px_size), np.arange(0, max_y+px_size, px_size), indexing='xy')
    px_coords = np.array([px_x, px_y]).transpose(1,2,0).reshape(-1,2)
    px_coords = np.ascontiguousarray(px_coords).astype(np.float32)
    pcd_coords = np.ascontiguousarray(pt_array[:,0:2]).astype(np.float32)

    # Find the nearest neighbor points of each pixel
    cpu_index = faiss.IndexFlatL2(2)
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    gpu_index.add(pcd_coords)
    nn_dist, nn_idx = gpu_index.search(px_coords, 1)

    # Get the depth of each pixel. depth=0 if no point nearby
    depth_list = pt_array[:,2][nn_idx]
    depth_list[nn_dist > px_size] = 0

    # Get depth image
    w, h = int(np.ceil(max_x/px_size)+1), int(np.ceil(max_y/px_size)+1)
    depth_mat = depth_list.reshape(h,w)
    depth_img = np.zeros((h,w), np.uint8)
    depth_img = cv2.normalize(depth_mat, depth_img, 0, 255, cv2.NORM_MINMAX)
    depth_img = depth_img.astype(np.uint8)

    # Rotate the sherd
    mask_img = np.zeros((h,w), np.uint8)
    mask_img[depth_mat > 0] = 255
    contours, _ = cv2.findContours(mask_img.copy(), 1, 1)
    contour = max(contours, key = cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    _, (w, h), _ = rect
    w, h = int(np.ceil(w)),int(np.ceil(h)) # w and h after rotation

    src_box = cv2.boxPoints(rect)
    dst_box = np.array([[0,h-1], [0,0], [w-1,0], [w-1,h-1]])
    rot_mat = cv2.getPerspectiveTransform(src_box.astype(np.float32), dst_box.astype(np.float32))
    rot_depth_img = cv2.warpPerspective(depth_img, rot_mat, (w,h))
    rot_depth_mat = cv2.warpPerspective(depth_mat, rot_mat, (w,h))
    rot_mask_img = cv2.warpPerspective(mask_img, rot_mat, (w,h))
    rot_mask_img[rot_mask_img < 128] = 0
    rot_mask_img[rot_mask_img >= 128] = 255

    # Visualization
    # color_depth_img = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2RGB)
    # color_depth_img = cv2.drawContours(color_depth_img.copy(), [np.int0(src_box)], 0, (0,0,255), 3)
    # cv2.imshow('depth_img', depth_img)
    # cv2.imshow('rot_depth_img', rot_depth_img)
    # cv2.imshow('color_depth_img', color_depth_img)
    # cv2.waitKey()

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
    clahe_depth_img = clahe.apply(rot_depth_img)

    return rot_depth_mat, clahe_depth_img, rot_mask_img
