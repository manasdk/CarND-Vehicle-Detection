import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog


class FeatureExtractor:

    @staticmethod
    def extract_features(img):
        pass

    @staticmethod
    def _get_bin_spatial_features(img, size=(32, 32)):
        """
        Returns the binned color feature of an image
        """
        return cv2.resize(img, size).ravel()

    @staticmethod
    def _get_color_hist_features(img, nbins=32, bins_range=(0, 256)):
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
    def _get_hog_features(img, orient, pix_per_cell=8, cell_per_block=2, feature_vec=True):
        hog_features = []
        # extract hog features for all channel
        for channel in range(img.shape[2]):
            features = hog(
                img[:, :, channel], orientations=orient,
                pixels_per_cell=(pix_per_cell, pix_per_cell),
                cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                visualise=False, feature_vector=feature_vec
            )
            hog_features.append(features)
        return np.ravel(hog_features)

    @staticmethod
    def _adjust_img_color_space(img, cspace='RGB'):
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
        img = FeatureExtractor._adjust_img_color_space(img, cspace=cspace)
        hog_features = FeatureExtractor._get_hog_features(img, orient, pix_per_cell, cell_per_block)
        color_hist_features = FeatureExtractor._get_color_hist_features(img)
        bin_spatial_features = FeatureExtractor._get_bin_spatial_features(img)
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
