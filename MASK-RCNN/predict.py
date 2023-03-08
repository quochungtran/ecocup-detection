#!/usr/bin/env python

import os
import sys
import resource
import numpy as np
import helper
import matplotlib.pyplot as plt
from importlib import reload
from rich import print

#%%
ROOT_DIR = os.path.abspath("./")

# allow python to search for Mask_RCNN package
MASK_RCNN_DIR = os.path.join(ROOT_DIR, "Mask_RCNN")
sys.path.append(MASK_RCNN_DIR)
from mrcnn import utils, config, model, visualize

class PredictionConfig(config.Config):
    NAME = "ecocup_cfg"
    NUM_CLASSES = 1 + 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


cfg = PredictionConfig()
model = model.MaskRCNN(mode='inference', model_dir="./", config=cfg)
model.load_weights("./ecocup_cfg20220611T0027/mask_rcnn_ecocup_cfg_0005.h5", by_name=True)
