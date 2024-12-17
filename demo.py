import cv2
import os
import numpy as np

import utility as util
import base
import axpansion as axp

input_dir = 'Input'
output_dir = 'Output'
tsukuba = os.path.join(input_dir, 'tsukuba')
map_dir = os.path.join(input_dir, 'map')

def demo_custom(folder_dir, max_offset, disparity_norm, model=1, size=3, unique=False , occlusion=10000):
    """
    Performs custom demo. Folder directory must contain an im0.png (left image) and im1.png (right image). Model 1 is the window algorithm
    which will use the other parameters. Model 2 will be alpha expansion which only uses max_offset and disparity_norm.
    """
    sample, ref = util.load_imgs(folder_dir)
    if model == 1:
        map = base.perform_match(ref, sample, max_offset, disparity_norm, size, unique, occlusion)
    else:
        map = axp.alpha_expansion(ref, sample, max_offset, disparity_norm)
    cv2.imwrite(os.path.join(output_dir, 'custom_demo.png'), map)
    cv2.imshow('sample', sample)
    cv2.imshow('reference', ref)
    cv2.imshow('disparity map', map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demo_window():
    sample, ref = util.load_imgs(tsukuba)
    map = base.perform_match(ref, sample,  max_offset=24, disparity_norm=16, size=9)
    cv2.imwrite(os.path.join(output_dir, 'tsukuba_window.png'), map)

    map = base.perform_match(ref, sample, max_offset=24, disparity_norm=16, size=9, unique=True, occlusion=10000)
    cv2.imwrite(os.path.join(output_dir, 'tsukuba_window_unique.png'), map)

    rev_sample = ref[::, ::-1]
    rev_ref = sample[::, ::-1]
    rev_map = base.perform_match(rev_ref, rev_sample, max_offset=24, disparity_norm=16, size=9)
    rev_map = rev_map[::, ::-1]
    cv2.imwrite(os.path.join(output_dir, 'tsukuba_window_rev.png'), rev_map)

    shape = ref.shape
    canvas = np.zeros(shape).astype(np.uint8)
    canvas[:, :shape[1] - 8] = rev_map[:, 8:]
    matches = map == canvas
    occlu = np.multiply(map, matches)
    cv2.imwrite(os.path.join(output_dir, 'tsukuba_window_occlusion.png'), occlu)

    sample, ref = util.load_imgs(map_dir)
    map = base.perform_match(ref, sample, max_offset=8, disparity_norm=4, size=9)
    cv2.imwrite(os.path.join(output_dir, 'map_window.png'), map)

def demo_alpha_expansion():
    sample, ref = util.load_imgs(tsukuba)
    map = axp.alpha_expansion(ref, sample, max_offset=24, disparity_norm=16)
    cv2.imwrite(os.path.join(output_dir, 'tsukuba_axpansion.png'), map)

    sample, ref = util.load_imgs(map_dir)
    map = axp.alpha_expansion(ref, sample, max_offset=8, disparity_norm=4)
    cv2.imwrite(os.path.join(output_dir, 'map_axpansion.png'), map)

def demo_comparisons():
    print('Tsukuba Dataset:')
    graph_cut = cv2.imread(os.path.join('final_samples', 'tsukuba_axpansion.png'))
    window = cv2.imread(os.path.join('final_samples', 'tsukuba_window.png'))
    window_occlusion = cv2.imread(os.path.join('final_samples', 'tsukuba_window_occlusion.png'))
    window_unique = cv2.imread(os.path.join('final_samples', 'tsukuba_window_unique.png'))
    ground = cv2.imread(os.path.join(input_dir, 'tsukuba', 'ground.png'))

    shape = graph_cut.shape
    border = 18
    canvas = np.zeros(shape).astype(np.uint8)

    canvas[border:shape[0] - border, border:shape[1] - border] = graph_cut[border:shape[0] - border, border:shape[1] - border]
    compare = util.compare_disparity(ground, canvas)
    print('Alpha expansion MSE: ', compare)

    canvas[border:shape[0] - border, border:shape[1] - border] = window[border:shape[0] - border, border:shape[1] - border]
    compare = util.compare_disparity(ground, canvas)
    print('SSD MSE: ', compare)

    canvas[border:shape[0] - border, border:shape[1] - border] = window_occlusion[border:shape[0] - border, border:shape[1] - border]
    compare = util.compare_disparity(ground, canvas)
    print('SSD corrected MSE: ', compare)

    canvas[border:shape[0] - border, border:shape[1] - border] = window_unique[border:shape[0] - border, border:shape[1] - border]
    compare = util.compare_disparity(ground, canvas)
    print('SSD Unique MSE: ', compare)

    print('Map Dataset:')
    graph_cut = cv2.imread(os.path.join('final_samples', 'map_axpansion.png'))
    window = cv2.imread(os.path.join('final_samples', 'map_window.png'))
    ground = cv2.imread(os.path.join(input_dir, 'map', 'ground.png'))

    compare = util.compare_disparity(ground, graph_cut)
    print('Alpha expansion MSE: ', compare)

    compare = util.compare_disparity(ground, window)
    print('SSD MSE: ', compare)

if __name__ == "__main__":
    # Example for running a demo for your own images:
    # demo_custom(folder_dir=os.path.join('Input', 'tsukuba'), max_offset=24, disparity_norm=16, model=1, size=9, unique=False, occlusion=10000)
    demo_window()
    demo_alpha_expansion()
    demo_comparisons()