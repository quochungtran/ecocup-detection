#!/usr/bin/env python3

# draw an image with detected objects
from xml.etree import ElementTree
import matplotlib.pyplot as plt
import os, sys


def draw_box_around_object(filename, boxes_list):
    """Draw image with boxes around the detected object

    Args:
        filename (str): path of image
        boxes_list (tuple): coordinations of the box
    """
    data = plt.imread(filename)
    plt.imshow(data)
    ax = plt.gca()

    # plot each box
    for box in boxes_list:
        y1, x1, y2, x2 = box
        width, height = x2 - x1, y2 - y1

        rect = plt.Rectangle((x1, y1), width, height, fill=False, color="red")
        ax.add_patch(rect)

    plt.show()


def extract_boxes(name: str):
    tree = ElementTree.parse(source=name)
    root = tree.getroot()

    boxes = []
    for box in root.findall(".//bndbox"):
        xmin = int(box.find("xmin").text)
        ymin = int(box.find("ymin").text)
        xmax = int(box.find("xmax").text)
        ymax = int(box.find("ymax").text)
        coors = [ymin, xmin, ymax, xmax]
        boxes.append(coors)

    width = int(root.find(".//size/width").text)
    height = int(root.find(".//size/height").text)
    return boxes, width, height


def plot_first_nine(dataset):
    # define subplot
    for i in range(9):
        plt.subplot(330 + 1 + i)
        # turn off axis labels
        plt.axis("off")
        # plot raw pixel data
        image = dataset.load_image(i)
        plt.imshow(image)
        # plot all masks
        mask, _ = dataset.load_mask(i)
        for j in range(mask.shape[2]):
            plt.imshow(mask[:, :, j], cmap="gray", alpha=0.3)

    plt.plot()


ROOT_DIR = os.path.abspath("./")

# allow python to search for Mask_RCNN package
MASK_RCNN_DIR = os.path.join(ROOT_DIR, "Mask_RCNN")
sys.path.append(MASK_RCNN_DIR)
from mrcnn import utils, config, model, visualize


def plot_image(train_set, image_id):
    # load the image
    image = train_set.load_image(image_id)
    # load the masks and the class ids
    mask, class_ids = train_set.load_mask(image_id)
    # extract bounding boxes from the masks
    bbox = utils.extract_bboxes(mask)
    # display image with masks and bounding boxes
    visualize.display_instances(image, bbox, mask, class_ids, train_set.class_names)