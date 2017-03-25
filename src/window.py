import os
import time

import cv2
import numpy as np
from scipy.ndimage.measurements import label

from classifier import ClassifierTrainer
from features import HogFeaturesExtractor, FeatureExtractor, print_feature_info
from heatmap import Heatmapper

MODULE_DIR = os.path.dirname(__file__)


class SubSampleWindowSearch:

    def __init__(self):
        # assuming this is a trained classifier
        self.classifier, _, self.feature_scaler = ClassifierTrainer.load_classifier()

    def find_cars(
        self, img, ystart, ystop, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins
    ):
        """
        Optimized method for computing the hog features once and then using that computation to
        drive the vehicle detection.
        """
        draw_img = np.copy(img)

        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = FeatureExtractor.adjust_img_color_space(img_tosearch, cspace='YUV')

        # Define blocks and steps as above
        nxblocks = (ctrans_tosearch.shape[1] // pix_per_cell)-1
        nyblocks = (ctrans_tosearch.shape[0] // pix_per_cell)-1
        nfeat_per_block = orient*cell_per_block**2
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell)-1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hfe = HogFeaturesExtractor(ctrans_tosearch)
        hfe.run_hog_extraction(orient, pix_per_cell, cell_per_block, feature_vec=False)

        bbox_list = []

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this sub img
                hog_features = hfe.get_partial_hog_feature(
                    (ypos, ypos + nblocks_per_window), (xpos, xpos + nblocks_per_window)
                )
                # print_feature_info("hog", hog_features)

                xleft = xpos*pix_per_cell
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get color features
                spatial_features = FeatureExtractor.get_bin_spatial_features(
                    subimg, size=spatial_size
                )
                # print_feature_info("spatial_features", spatial_features)

                # Get histogram features
                hist_features = FeatureExtractor.get_color_hist_features(
                    subimg, nbins=hist_bins
                )
                # print_feature_info("hist_features", hist_features)

                # Scale features and make a prediction
                test_features = np.hstack((hog_features, hist_features, spatial_features))
                test_features = self.feature_scaler.transform(test_features)
                test_prediction = self.classifier.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft)
                    ytop_draw = np.int(ytop)
                    win_draw = np.int(window)
                    bbox_list.append((
                        (xbox_left, ytop_draw + ystart),
                        (xbox_left + win_draw, ytop_draw + win_draw+ystart)
                    ))
        draw_img = self._draw_boxes(draw_img, bbox_list)
        return draw_img

    def find_cars_in_windows_scaled(
        self, img, cspace='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8,
        cell_per_block=2
    ):
        test_windows = []
        window_inputs = [
            ([400, 500], (64, 64)),
            ([400, 600], (80, 80)),
            ([400, 680], (128, 128)),
            ([400, 680], (140, 140))
        ]
        for y_start_stop, xy_window in window_inputs:
            test_windows.extend(self._slide_window(
                img, x_start_stop=[None, None], y_start_stop=y_start_stop, xy_window=xy_window
            ))
        return self._find_cars_in_windows(
            img=img, windows=test_windows, cspace=cspace, spatial_size=spatial_size,
            hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
            cell_per_block=cell_per_block
        )

    def _find_cars_in_windows(
        self, img, windows, cspace='RGB', spatial_size=(32, 32), hist_bins=32,
        orient=9, pix_per_cell=8, cell_per_block=2
    ):
        draw_img = np.copy(img)
        #1) Create an empty list to receive positive detection windows
        bbox_list = []
        #2) Iterate over all windows in the list
        for window in windows:
            #3) Extract the test window from original image
            test_img = cv2.resize(
                img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64,64)
            )
            #4) Extract features for that window
            features = self._single_img_features(
                img=test_img, cspace=cspace, orient=orient, pix_per_cell=pix_per_cell,
                cell_per_block=cell_per_block
            )
            #5) Scale extracted features to be fed to classifier
            test_features = self.feature_scaler.transform(np.array(features).reshape(1, -1))
            #6) Predict using your classifier
            prediction = self.classifier.predict(test_features)
            #7) If positive (prediction == 1) then save the window
            if prediction == 1:
                bbox_list.append(window)
        #8) Return windows for positive detections
        draw_img = self._draw_boxes(draw_img, bbox_list)
        return draw_img

    def _slide_window(
        self, img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64),
        xy_overlap=(0.5, 0.5)
    ):
        # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
        ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
        nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
        ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]

                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    def _single_img_features(
        self, img, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2
    ):
        return FeatureExtractor.extract_features_for_img(
            img=img, cspace=cspace, orient=orient, pix_per_cell=pix_per_cell,
            cell_per_block=cell_per_block
        )

    def _draw_boxes(self, img, bbox_list):
        # 1. create a heatmap of detections
        heatmapper = Heatmapper(img)
        heatmap = heatmapper.add_heat(bbox_list=bbox_list, threshold=1)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)

        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 2)

        # Return the image
        return img


def do_window_search(rel_in_path, rel_out_path):
    t = time.time()
    window_search = SubSampleWindowSearch()
    in_path = os.path.join(MODULE_DIR, rel_in_path)
    out_path = os.path.join(MODULE_DIR, rel_out_path)
    in_img = cv2.imread(in_path)
    out_img = window_search.find_cars(
        img=in_img, ystart=400, ystop=656, orient=9, pix_per_cell=8, cell_per_block=2,
        spatial_size=(32, 32), hist_bins=32
    )
    cv2.imwrite(out_path, out_img)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to find cars')


def do_window_search_2(rel_in_path, rel_out_path):
    t = time.time()
    window_search = SubSampleWindowSearch()
    in_path = os.path.join(MODULE_DIR, rel_in_path)
    out_path = os.path.join(MODULE_DIR, rel_out_path)
    in_img = cv2.imread(in_path)
    out_img = window_search.find_cars_in_windows_scaled(
        img=in_img, cspace='YUV', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8,
        cell_per_block=2
    )
    cv2.imwrite(out_path, out_img)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to find cars')


if __name__ == '__main__':
    do_window_search_2('../test_images/test1.jpg', '../output_images/test1_heat2.jpg')
    do_window_search_2('../test_images/test2.jpg', '../output_images/test2_heat2.jpg')
    do_window_search_2('../test_images/test3.jpg', '../output_images/test3_heat2.jpg')
    do_window_search_2('../test_images/test4.jpg', '../output_images/test4_heat2.jpg')
    do_window_search_2('../test_images/test5.jpg', '../output_images/test5_heat2.jpg')
    do_window_search_2('../test_images/test6.jpg', '../output_images/test6_heat2.jpg')
