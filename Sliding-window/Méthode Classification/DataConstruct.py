import os
from operator import itemgetter
from random import randrange

import cv2 as cv
import numpy as np
from skimage import io


class DataConstruct():
    xSize = 24
    ySize = 24

    # Fonction imaginée pour retrouver le ratio le plus utilisé avec une marge donnée (ici 0.1) afin d'optimiser notre fenêtre englobante
    def find_best_dimensions(self):
        data = np.genfromtxt("combined-csv-files.csv", delimiter=",",
                             names=["y", "x", "h", "w", "isDifficult"], invalid_raise=False)
        res = {}
        for i in data:
            ratioI = i["h"]/i["w"]
            res[ratioI] = 0
            for j in data:
                ratioJ = j["h"]/j["w"]
                if ratioJ > ratioI - 0.1 and ratioJ < ratioI + 0.1:
                    res[ratioI] += 1

        sorted_res = sorted(res.items(), key=itemgetter(1), reverse=True)


    # Génère tous les échantillons
    def generate_all(self):
        self.generate_pos_train()
        self.generate_neg_train()
        self.generate_pos_train_p21()
        self.generate_neg_train_p21()

    # Découpe les boîtes positives à partir des images et annotations fournies dans dataset-main/train
    def generate_pos_train(self):
        index = 0

        for image_path in os.listdir("dataset-main/train/images/pos"):
            im = io.imread("dataset-main/train/images/pos/%s" % image_path)

            base_name = image_path.split('.')[0]
            try:
                data = np.genfromtxt("dataset-main/train/labels_csv/%s.csv" % base_name,
                                     delimiter=",", names=["y", "x", "h", "w", "isDifficult"], invalid_raise=False)
                if data.size == 1:
                    data = np.array([data])
                for d in data:
                    im_res = self.im_transform(im, d['y'].astype('int'), d['x'].astype(
                        'int'), d['h'].astype('int'), d['w'].astype('int'))
                    cv.imwrite(
                        "dataset-main/train/resized_images/pos/%d.jpg" % index, im_res)
                    index += 1
            except IOError:
                continue

    # Découpe 10 sous images aléatoires de dimensions désirées pour chaque image négative fournie dans dataset-main/train
    def generate_neg_train(self):
        index = 0
        for image_path in os.listdir("dataset-main/train/images/neg"):
            im = io.imread("dataset-main/train/images/neg/%s" % image_path)
            for i in range(0, 10):
                x = randrange(0, im.shape[1]-self.xSize)
                y = randrange(0, im.shape[0]-self.ySize)

                im_res = self.im_transform(im, y, x, self.ySize, self.xSize)
                cv.imwrite(
                    "dataset-main/train/resized_images/neg/%d.jpg" % index, im_res)
                index += 1

    # Découpe les boîtes positives à partir des images et annotations fournies dans dataset-main/train_p21
    def generate_pos_train_p21(self):
        index = 0
        for image_path in os.listdir("dataset-main/train_p21/images/pos"):
            im = io.imread("dataset-main/train_p21/images/pos/%s" % image_path)

            base_name = image_path.split('.')[0]
            try:
                data = np.genfromtxt("dataset-main/train_p21/labels_csv/%s.csv" % base_name,
                                     delimiter=",", names=["y", "x", "h", "w", "isDifficult"], invalid_raise=False)
                if data.size == 1:
                    data = np.array([data])
                for d in data:
                    im_res = self.im_transform(im, d['y'].astype('int'), d['x'].astype(
                        'int'), d['h'].astype('int'), d['w'].astype('int'))
                    cv.imwrite(
                        "dataset-main/train_p21/resized_images/pos/%d.jpg" % index, im_res)
                    index += 1

            except IOError:
                continue

    # Découpe 10 sous images aléatoires de dimensions désirées pour chaque image négative fournie dans dataset-main/train_p21
    def generate_neg_train_p21(self):
        index = 0
        for image_path in os.listdir("dataset-main/train_p21/images/neg"):
            im = io.imread("dataset-main/train_p21/images/neg/%s" % image_path)
            for i in range(0, 10):
                x = randrange(0, im.shape[1]-self.xSize)
                y = randrange(0, im.shape[0]-self.ySize)

                im_res = self.im_transform(im, y, x, self.ySize, self.xSize)
                cv.imwrite(
                    "dataset-main/train_p21/resized_images/neg/%d.jpg" % index, im_res)
                index += 1

     # Forme une image au ratio désiré à partir d'une image et de coordonnées d'une boîte
    def im_transform(self, im, y, x, h, w):
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        im = im[y:y+h, x:x+w]
        im = cv.resize(im, (self.xSize, self.ySize))
        return im
