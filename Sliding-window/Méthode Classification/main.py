import itertools as it
import os

import cv2 as cv
import numpy as np
from numpy.lib import recfunctions
from skimage import io
from skimage.feature import hog
from sklearn import svm

from ClassifierFeeder import ClassifierFeeder
from DataConstruct import DataConstruct

xSize = DataConstruct.xSize
ySize = DataConstruct.ySize

# On récupère un classifier de type SVC qu'on entraine sur nos images
feeder = ClassifierFeeder()
feeder.populate()
clf = feeder.train(svm.SVC())


# Retourne l'interséction sur union de deux boîtes
def IoU(bb1, bb2):
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


# Filtre les résultats pour ne retourner que le meilleur score lorsque l'IoU de deux boîtes est supérieure à un seuil (ici 0.5)
def filter_results(results):
    start = 0
    nb = 1

    mask = np.ones(len(results))

    for i in range(0, len(results)-1):
        x = results[i][0]
        if results[i+1][0] == x and i+2 != len(results):
            nb += 1
        else:
            comb = it.combinations(range(start, start+nb), 2)
            for cb in comb:
                bb1 = {'x1': results[cb[0]][2], 'y1': results[cb[0]][1], 'x2': results[cb[0]][2] +
                       results[cb[0]][4], 'y2': results[cb[0]][1]+results[cb[0]][3], 's': results[cb[0]][5]}
                bb2 = {'x1': results[cb[1]][2], 'y1': results[cb[1]][1], 'x2': results[cb[1]][2] +
                       results[cb[1]][4], 'y2': results[cb[1]][1]+results[cb[1]][3], 's': results[cb[1]][5]}

                if IoU(bb1, bb2) > 0.5:
                    if bb1['s'] > bb2['s']:
                        mask[cb[1]] = 0
                    else:
                        mask[cb[0]] = 0

            start = i+1
            nb = 1
    print("%d results filtered" % np.count_nonzero(mask == 0))
    return results[mask.astype(bool)]


filtered_results_csv = ""

vp = 0
fp = 0
fn = 0
nb_treated = 0

# Pour chaque image, on prend des fenêtres décalées d'un certain pas, et on utilise le classifier pour déterminer si elles contiennent des écocups.
# Après ça, on recommence sur l'image downscalée.
# Enfin, on filtre les boîtes qui ont un IoU > 0.5.
for image_path in os.listdir("dataset-main/train_p21/images/pos"):
    pos_results = []

    im = io.imread("dataset-main/train_p21/images/pos/%s" % image_path)
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    scale_step = 0.05
    scale = 1 + scale_step
    while scale > 0.05:
        scale -= scale_step
        new_width = int(im.shape[1] * scale)
        new_height = int(im.shape[0] * scale)
        resized_im = cv.resize(im, (new_width, new_height),
                               interpolation=cv.INTER_AREA)
        for i in range(0, resized_im.shape[0]-ySize, 5):
            for j in range(0, resized_im.shape[1]-xSize, 5):
                box = resized_im[i:i+ySize, j:j+xSize]
                res = clf.decision_function([hog(box)])
                if res > 0:
                    pos_results.append(
                        (image_path, i/scale, j/scale, xSize/scale, ySize/scale, res))
    filtered_results = filter_results(np.array(pos_results, dtype=object))

    dt = np.dtype([('used', '<f8')])
    base_name = image_path.split('.')[0]
    data = np.genfromtxt("dataset-main/train_p21/labels_csv/%s.csv" % base_name,
                         delimiter=",", names=["y", "x", "h", "w", "isDifficult"], invalid_raise=False)
    if data.size == 1:
        data = np.array([data])
    new_column = np.zeros(len(data), dtype=dt)
    data = recfunctions.merge_arrays((data, new_column), flatten=True)

    # Finalement, on compare nos résultats à la vérité, pour pouvoir connaitre les performances de notre détecteur.
    for result in filtered_results:
        filtered_results_csv += "%s,%d,%d,%d,%d,%f\n" % (
            result[0], result[1], result[2], result[3], result[4], result[5])

        test = False
        for d in data:
            if not d['used'].astype(bool):
                bb1 = {'x1': result[2], 'y1': result[1],
                       'x2': result[2]+result[4], 'y2': result[1]+result[3]}
                bb2 = {'x1': d['x'].astype('int'), 'y1': d['y'].astype('int'), 'x2': d['x'].astype(
                    'int')+d['w'].astype('int'), 'y2': d['y'].astype('int')+d['h'].astype('int')}
                if IoU(bb1, bb2) > 0.5:
                    test = True
                    vp += 1
                    d['used'] = 1
                    break
        if test == False:
            fp += 1
    fn += np.count_nonzero(data['used'] == 0)
    nb_treated += 1
    print("ended image %s, %d/721" % (image_path, nb_treated))


p = vp/(vp+fp)
r = vp/(vp+fn)

print("vp: %d, fp: %d, fn: %d" % (vp, fp, fn))
print("précision: %f" % p)
print("rappel: %f" % r)
if p+r != 0:
    f1 = 2*(p*r/(p+r))
    print("f1: %f" % f1)

with open('results.csv', 'w') as out:
    out.write(filtered_results_csv)
