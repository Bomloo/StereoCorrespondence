import cv2
import numpy as np
from numba import njit, prange

import utility as util

def apply_border(img, border_len, border_type):
    return cv2.copyMakeBorder(img, border_len, border_len, border_len, border_len, border_type)

@njit
def window_match(ref, sample, size, max_offset):
    change = size//2
    m = ref.shape[0] - 2 * change
    n = ref.shape[1] - 2 * change

    ref_features = []
    sample_features = []
    disparity = []
    occluded = []

    for i in prange(m):
        k_val=0
        for j in prange(n):
            min_ssd = 255**2 * (1 + 2 * change) * (1 + 2 * change)
            for k in prange(0, max_offset):
                k_mod = j+k
                if k_mod < n:
                    ssd = 0
                    for l in prange(1 + 2 * change):
                        for o in prange(1 + 2 * change):
                            diff = np.float64(ref[i + l][j + o]) - np.float64(sample[i + l][k_mod + o])
                            ssd = ssd + diff * diff
                    # minimum ssd
                    if ssd < min_ssd:
                        min_ssd = ssd
                        k_val = k_mod
            ref_features.append((j, i))
            sample_features.append((k_val, i))
            dis = k_val - j
            if dis < 0:
                dis = -1*dis
            disparity.append(dis)
        occluded.append((-1, -1))
    return ref_features, sample_features, disparity, occluded

@njit
def unique_window_match(ref, sample, size, max_offset, occlusion):
    change = size//2
    m = ref.shape[0] - 2 * change
    n = ref.shape[1] - 2 * change

    ref_features = []
    sample_features = []
    no_dupes = {}
    disparity = []
    occluded = []

    for i in prange(m):
        k_val=0
        for j in prange(n):
            matched = False
            min_ssd = 255**2 * (1 + 2 * change) * (1 + 2 * change)
            for k in prange(0, max_offset):
                k_mod = j+k
                if k_mod < n and (k_mod, i) not in no_dupes:
                    ssd = 0
                    for l in prange(1 + 2 * change):
                        for o in prange(1 + 2 * change):
                            diff = np.float64(ref[i + l][j + o]) - np.float64(sample[i + l][k_mod + o])
                            ssd = ssd + diff * diff
                    # minimum ssd
                    if ssd < min_ssd and ssd < occlusion:
                        min_ssd = ssd
                        k_val = k_mod
                        matched = True
            if matched:
                ref_features.append((j, i))
                sample_features.append((k_val, i))
                no_dupes[(k_val, i)] = 1
                dis = k_val - j
                if dis < 0:
                    dis = -1*dis
                disparity.append(dis)
            else:
                occluded.append((j, i))
    return ref_features, sample_features, disparity, occluded

def perform_match(ref, sample, max_offset, disparity_norm, size=3, unique=False, occlusion=1000):
    print('Performing SSD window match')
    shape = ref.shape
    change = size//2
    ref = apply_border(ref, change, cv2.BORDER_CONSTANT)
    sample = apply_border(sample, change, cv2.BORDER_CONSTANT)
    if unique:
        ref_features, sample_features, disparity, occluded = unique_window_match(ref, sample, size, max_offset, occlusion)
    else:
        ref_features, sample_features, disparity, occluded = window_match(ref, sample, size, max_offset)
        occluded = occluded[1:]
    print('Matching finished')
    map = util.generate_map(shape, ref_features, disparity, disparity_norm)
    return map