import os

from skimage import io
from skimage.feature import hog

class ClassifierFeeder():
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    def populate(self):
        for image_path in os.listdir("dataset-main/train/resized_images/pos"):
            I = io.imread(
                "dataset-main/train/resized_images/pos/%s" % image_path)
            I = hog(I)
            self.X_train.append(I)
            self.y_train.append(1)

        for image_path in os.listdir("dataset-main/train/resized_images/neg"):
            I = io.imread(
                "dataset-main/train/resized_images/neg/%s" % image_path)
            I = hog(I)
            self.X_train.append(I)
            self.y_train.append(-1)

        for image_path in os.listdir("dataset-main/train_p21/resized_images/pos"):
            I = io.imread(
                "dataset-main/train_p21/resized_images/pos/%s" % image_path)
            I = hog(I)
            self.X_test.append(I)
            self.y_test.append(1)

        for image_path in os.listdir("dataset-main/train_p21/resized_images/neg"):
            I = io.imread(
                "dataset-main/train_p21/resized_images/neg/%s" % image_path)
            I = hog(I)
            self.X_test.append(I)
            self.y_test.append(-1)

    def train(self, clf):
        clf.fit(self.X_train, self.y_train)
        print("{} : {}".format(type(clf), 1-clf.score(self.X_test, self.y_test)))
        return clf
