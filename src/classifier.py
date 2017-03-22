import cPickle
import glob
import os
import time

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

from features import FeatureExtractor

MODULE_DIR = os.path.dirname(__file__)
SAVED_CLASSIFIER_DATA = os.path.join(MODULE_DIR, 'classifier.dat')


class ClassifierTrainer:

    def __init__(self):
        self.classifier, self.is_trained = ClassifierTrainer.load_classifier()

    def train_classifier(self):
        """
        Train the vehicle identificaton classifier using vehicle and non-vehicle
        training data
        """
        features, labels = ClassifierTrainer._get_training_data()
        # create train and test sets
        rand_state = np.random.randint(0, 100)
        features_train, features_test, labels_train, labels_test = train_test_split(
            features, labels, test_size=0.2, random_state=rand_state)
        if self.is_trained:
            return self.classifier, features_train, features_test, labels_train, labels_test
        print('Started classifier training')
        t=time.time()
        self.classifier.fit(features_train, labels_train)
        t2=time.time()
        print(round(t2-t, 2), 'Seconds to train classifier...')
        self.is_trained = True
        return self.classifier, features_train, features_test, labels_train, labels_test

    def train_classifier_and_report_score(self):
        """
        Train the vehicle identificaton classifier using vehicle and non-vehicle
        training data. It also prints the score of the classifier
        """
        classifier, _, features_test, _, labels_test = self.train_classifier()
        print 'classifier accuracy : %s' % classifier.score(features_test, labels_test)

    def save_classifier(self, classifier_path=SAVED_CLASSIFIER_DATA):
        if not self.is_trained:
            raise Exception('cannot save untrained classifier')
        with open(classifier_path, 'wb') as classifier_out:
            cPickle.dump(self.classifier, classifier_out)

    @staticmethod
    def load_classifier(classifier_path=SAVED_CLASSIFIER_DATA):
        if not os.path.exists(classifier_path):
            return LinearSVC(), False
        print('loading saved classifier')
        with open(classifier_path, 'rb') as classifier_out:
            return cPickle.load(classifier_out), True

    @staticmethod
    def _get_training_data():

        print('Extracting vehicle features ...')
        t = time.time()
        vehicle_features = FeatureExtractor.extract_features_for_multiple_images(
            ClassifierTrainer._get_vehicle_img_paths()
        )
        t2 = time.time()
        print round(t2-t, 2), 'Seconds to extract vehicle features'

        print('Extracting non-vehicle features ...')
        t = time.time()
        non_vehicle_features = FeatureExtractor.extract_features_for_multiple_images(
            ClassifierTrainer._get_non_vehicle_img_paths()
        )
        t2 = time.time()
        print round(t2-t, 2), 'Seconds to extract non-vehicle features'

        # combine an scale the vehicle and non-vehicle features
        combined = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)
        # Fit a per-column scaler
        feature_scaler = StandardScaler().fit(combined)
        # Apply the scaler to X
        features = feature_scaler.transform(combined)

        # Define the labels vector
        labels = np.hstack(
            (np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features)))
        )

        return features, labels

    @staticmethod
    def _get_vehicle_img_paths():
        pattern = os.path.join(MODULE_DIR, '../train_data/vehicles/*/*.png')
        return glob.glob(pattern)

    @staticmethod
    def _get_non_vehicle_img_paths():
        pattern = os.path.join(MODULE_DIR, '../train_data/non-vehicles/*/*.png')
        return glob.glob(pattern)

if __name__ == '__main__':
    trainer = ClassifierTrainer()
    trainer.train_classifier_and_report_score()
    trainer.save_classifier()
