#!/usr/bin/env python3

#%%
import os
import sys
import resource
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
from rich import print

#%%
ROOT_DIR = os.path.abspath("./")

# allow python to search for Mask_RCNN package
MASK_RCNN_DIR = os.path.join(ROOT_DIR, "Mask_RCNN")
sys.path.append(MASK_RCNN_DIR)
from mrcnn import utils, config, model, visualize
import helper

# %%
class EcoCupDataset(utils.Dataset):
    def load_dataset(self, dataset_dirs, is_train: bool) -> None:
        self.add_class("dataset", 1, "ecocup")

        for dataset_dir in dataset_dirs:
            labels_dir = os.path.join(dataset_dir, "labels")
            images_dir = os.path.join(dataset_dir, "images/pos")

            images_files = os.listdir(images_dir)
            images_split_pos = int(0.8 * len(images_files))
            images_files = (
                images_files[:images_split_pos]  # first 80% for training
                if is_train
                else images_files[images_split_pos:]  # last 20% for testing
            )

            for filename in images_files:
                # extract image id (e.g. huarderi_pos_001.jpg to huarderi_pos_001)
                image_id = os.path.splitext(os.path.basename(filename))[0]
                img_path = os.path.join(images_dir, filename)

                label_id = image_id + ".xml"
                label_path = os.path.join(labels_dir, label_id)
                if not os.path.exists(label_path):
                    continue

                # add to dataset
                self.add_image(
                    "dataset", image_id=image_id, path=img_path, annotation=label_path
                )
        return

    def load_mask(self, image_id):
        image_metadata = self.image_info[image_id]
        label_path = image_metadata["annotation"]

        # load XML
        boxes, width, height = helper.extract_boxes(label_path)
        # create one array for all masks, each on a different channel
        masks = np.zeros(shape=(height, width, len(boxes)), dtype=np.uint8)

        # create masks
        class_ids = []
        for i, box in enumerate(boxes):
            row_begin, row_end = box[0], box[2]
            col_begin, col_end = box[1], box[3]
            masks[row_begin:row_end, col_begin:col_end, i] = 1
            class_ids.append(self.class_names.index("ecocup"))

        return masks, np.asarray(class_ids, dtype=np.int32)

    def image_reference(self, image_id):
        image_metadata = self.image_info[image_id]
        return image_metadata["path"]


# %%
# define a configuration for the model
class EcocupConfig(config.Config):
    NAME = "ecocup_cfg"
    # Number of classes (background + ecocup)
    NUM_CLASSES = 1 + 1
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 50


dataset_dirs = ["./train_p21", "./train_p22"]
train_set = EcoCupDataset()
train_set.load_dataset(dataset_dirs, is_train=True)
train_set.prepare()
print(f"Train: {len(train_set.image_ids)}")

# test/val set
test_set = EcoCupDataset()
test_set.load_dataset(dataset_dirs, is_train=False)
test_set.prepare()
print(f"Test: {len(test_set.image_ids)}")

# %%
ecocup_config = EcocupConfig()
model = model.MaskRCNN(mode="training", model_dir="./", config=ecocup_config)
# load weights (mscoco)
model.load_weights(
    "mask_rcnn_coco.h5",
    by_name=True,
    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"],
)

# %%
# train weights (output layers or "heads")
model.train(
    train_set, test_set, learning_rate=ecocup_config.LEARNING_RATE, epochs=10, layers="heads"
)
