import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog


class HogFeaturesExtractor:
    """
    Per image hog feature extraction.
    """
    def __init__(self, img):
        self.img = img
        self.hog_ch1 = None
        self.hog_ch2 = None
        self.hog_ch3 = None

    def run_hog_extraction(self, orient, pix_per_cell=8, cell_per_block=2, feature_vec=True):
        """
        Runs hog feature extraction on the entire image
        """
        hog_features = []
        for channel in range(self.img.shape[2]):
            features = hog(
                self.img[:, :, channel], orientations=orient,
                pixels_per_cell=(pix_per_cell, pix_per_cell),
                cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                visualise=False, feature_vector=feature_vec
            )
            hog_features.append(features)
        self.hog_ch1 = hog_features[0]
        self.hog_ch2 = hog_features[1]
        self.hog_ch3 = hog_features[2]

    def get_full_hog_features(self):
        """
        Flattened hog features of the entire image
        """
        return np.hstack((self.hog_ch1.ravel(), self.hog_ch2.ravel(), self.hog_ch3.ravel()))

    def get_partial_hog_feature(self, top_left, bottom_right):
        """
        Flattened hog features of part of the image as defined by the top_left and bottom_right
        corner points of the sub-image
        """
        return np.hstack((
            self.hog_ch1[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]].ravel(),
            self.hog_ch2[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]].ravel(),
            self.hog_ch3[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]].ravel()
        ))


class FeatureExtractor:
    """
    Entry point to Feature extraction methods
    """

    @staticmethod
    def extract_features(img):
        pass

    @staticmethod
    def get_bin_spatial_features(img, size=(32, 32)):
        """
        Returns the binned color feature of an image
        """
        return cv2.resize(img, size).ravel()

    @staticmethod
    def get_color_hist_features(img, nbins=32, bins_range=(0, 256)):
        """
        Returns the color histogram features of an image
        """
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
        channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
        channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        return hist_features

    @staticmethod
    def get_hog_features(img, orient, pix_per_cell=8, cell_per_block=2, feature_vec=True):
        hfe = HogFeaturesExtractor(img)
        hfe.run_hog_extraction(orient, pix_per_cell, cell_per_block, feature_vec)
        return hfe.get_full_hog_features()

    @staticmethod
    def adjust_img_color_space(img, cspace='RGB'):
        if cspace == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            # not sure if this copy is required
            img = np.copy(img)
        return img

    @staticmethod
    def extract_features(img_path, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2):
        img = cv2.imread(img_path)
        img = FeatureExtractor.adjust_img_color_space(img, cspace=cspace)
        hog_features = FeatureExtractor.get_hog_features(img, orient, pix_per_cell, cell_per_block)
        color_hist_features = FeatureExtractor.get_color_hist_features(img)
        bin_spatial_features = FeatureExtractor.get_bin_spatial_features(img)
        # combine features
        return np.concatenate((hog_features, color_hist_features, bin_spatial_features))

    @staticmethod
    def extract_features_for_multiple_images(
        img_paths, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2
    ):
        features = []
        for img_path in img_paths:
            per_img_features = FeatureExtractor.extract_features(
                img_path, cspace, orient, pix_per_cell, cell_per_block
            )
            features.append(per_img_features)
        return features


if __name__ == '__main__':
    """
    Simple unit test for FeaturesExtractor
    """
    # single image
    features = FeatureExtractor.extract_features(
        img_path='./train_data/vehicles/GTI_MiddleClose/image0476.png'
    )
    print len(features), features

    # multiple images
    img_paths=[
        './train_data/vehicles/GTI_MiddleClose/image0476.png',
        './train_data/vehicles/GTI_MiddleClose/image0475.png'
    ]
    features = FeatureExtractor.extract_features_for_multiple_images(img_paths=img_paths)
    print len(features), features
