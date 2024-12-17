import cv2
import os
import math
import numpy as np
from numba import njit, prange

def load_imgs(file_dir):
    """Load images.
    Args:
        file_dir (string): path to a folder containing rectified left and right images
        
    Returns:
        tuple: two-element tuple containing:
            img0 (numpy.array): Left view of a scene.
            img1 (numpy.array): Right view of a scene."""
    
    file0 = os.path.join(file_dir, 'im0.png')
    file1 = os.path.join(file_dir, 'im1.png')
    #left
    img0 = cv2.imread(file0)
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    #right
    img1 = cv2.imread(file1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    return img0, img1

@njit
def compare_disparity(ground_truth, disparity_map):
    """
    RMSE between ground_truth and disparity_map
    """
    total = disparity_map.shape[0] * disparity_map.shape[1]
    diff = ground_truth.astype(np.float64) - disparity_map.astype(np.float64)
    square = diff * diff
    rmse = math.sqrt(np.sum(square) / total)
    return rmse

@njit
def generate_map(shape, ref_features, disparity, normal=16):
    canvas = np.zeros(shape)

    disparity = np.array(disparity)
    max_dis = max(disparity)
    min_dis = min(disparity)
    dis_range = max_dis - min_dis
    disparity = normal * ((disparity - min_dis)/dis_range)
    disparity = disparity.astype(np.uint8)
    max_dis = max(disparity)
    min_dis = min(disparity)
    dis_range = max_dis - min_dis
    disparity_value = 255 * ((disparity - min_dis)/dis_range)
    
    for i in prange(len(ref_features)):
        x = ref_features[i][0]
        y = ref_features[i][1]
        canvas[y][x] = disparity_value[i]

    canvas = canvas.astype(np.uint8)
    return canvas