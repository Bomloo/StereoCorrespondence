# CS 6476 Final Project: Stereo Correspondence

# Packages
This project uses the following packages in its environment:

    Python 3.12.7
    PyMaxflow 1.2.13
    Numpy 1.26.4
    OpenCV 4.10.0
    Numba 0.60.0

# Folder and File Descriptions
    demo.py - contains code that will create disparity maps using the provided datasets and saves them to the Output folder

    axpansion.py - contains code that implements alpha expansion with graph cuts

    base.py - contains code that implements SSD window sliding feature matching

    utility.py - contains useful functions
    
    Input - Various folders containing image datasets for this project
        tsukuba - Contains files for the tsukuba dataset
            ground.png (384 x 288) - ground truth image
            im0.png (384 x 288) - left image
            im1.png (384 x 288) - right image
        map - Contains files for the map dataset
            ground.png (284 x 216) - ground truth image
            im0.png (284 x 216) - left image
            im1.png (284 x 216) - right image

    Output - Images produced by the demo will appear here

    final_samples - results obtained from experimentation and used in the report
        map_axpansion.png - using alpha expansion on the map dataset with max offset 8 and disparity normalized to 4
        map_window.png - using the SSD sliding window technique on the map dataset with max offset 8 and disparity normalized to 4
        tsukuba_axpansion.png - using alpha expansion on tsukuba dataset with max offset 24 and disparity normalized to 16
        tsukuba_window.png - using SSD sliding window technique on the tsukuba dataset with max offset 24 and disparity normalized to 16
        tsukuba_window_rev.png - using SSD sliding window technique on the tsukuba dataset but with the reference and sample images reversed (left to right)
        tsukuba_window_occlusion.png - compared right-to-left window disparity map with left-to-right window disparity map to get occluded or inaccurate pixels
        tsukuba_window_unique.png - using SSD sliding window technique on the tsukuba dataset with max offset 24, disparity normalized to 16, occlusion detection, and uniqueness preserved

# Using demo.py
    This file contains all the code to run a demo of the project. Run the file to produce disparity maps saved to the Output folder.

        demo_window() - runs demos of the SSD sliding window feature matching technique: tsukuba_window, tsukuba_window_unique, tsukuba_window_rev, tsukuba_window_occlusion, map_window

        demo_alpha_expansion() - runs demos of the alpha expansion algorithm: tsukuba_axpansion, map_axpansion

        demo_comparisons() - prints out comparisons between disparity maps and ground truth

        demo_custom() - creating a folder containing a left image as im0 and right image as im1 and giving this function the path along with parameters for a chosen model will save a disparity map in output. check function for parameter details

    NOTE: At the bottom, you can comment out demos you do not wish to run.
    NOTE: demo_comparisons() requires Input for the ground truth image and final_samples for the pre-saved disparity maps identical to demo. Can be modified to take current session results.

# Using axpansion.py
    This file contains all the code to run an alpha expansion algorithm but you only need to call alpha_expansion() to get a disparity map.

        alpha_expansion() requires:
            ref - a 2D numpy array that represents the right image (im1)
            sample - a 2D numpy array that represents the left image (im0)
            max_offset - the maximum pixel offset that an object in the scene will move
            disparity_norm - alpha_expansion calls generate_map in utility.py to build a disparity map and normalizes/reduces the disparity in the scene to this number
    
    NOTE: For the tsukuba dataset, this code runs for ~15 min, try to keep images small to avoid long runtimes. See demo for an example.

# Using base.py
    This file contains all the code to run a base implementation of SSD sliding window feature matching but you only need to call perform_match() to get a disparity map.

        perform_match() requires:
            ref - a 2D numpy array that represents the right image (im1)
            sample - a 2D numpy array that represents the left image (im0)
            max_offset - the maximum pixel offset that an object in the scene will move
            disparity_norm - alpha_expansion calls generate_map in utility.py to build a disparity map and normalizes/reduces the disparity in the scene to this number
            size - the size of the sliding window, default 3
            unique - detect occlusion and preserve uniqueness in matches, default False

    NOTE: For the tsukuba dataset, this code runs in seconds. Using unique gives very rough results, not recommended. See demo for an example.

# Using utility.py
    Contains various useful code that other functions will refer to. You may need to call functions from here to prepare images for running alpha_expansion() or perform_match().

        load_imgs() requires:
            file_dir - a path to a folder containing 2 images. im0.png will be the right image and im1.png will be the left image

        compare_disparity() requires:
            ground_truth - the ground_truth image, must be identical in size to disparity map and grayscale
            disparity_map - a grayscale disparity map

        generate_map() requires:
            shape - shape of the image
            ref_features - a list of tuples containing the x and y coordinates of a pixel, corresponds with disparity
            disparity - a list of pixel offset/disparity values that corresponds with ref_features
            normal - normalizes/reduces disparity to this number

    NOTE: load_imgs() will return the grayscale version of im0 and im1 in this order, it is important to set ref as im1 and sample as im0 for the algorithms to work. See demo for example.
    NOTE: compare_disparity() will return MSE. See demo for example.

    