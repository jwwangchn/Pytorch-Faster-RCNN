# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os

# root path
ROOT_PATH = os.path.abspath(r'/home/jwwangchn/Softwares/Pytorch-Faster-RCNN')

# dataloader
VOC_DATA_ROOT = '/home/jwwangchn/data/VOCdevkit'

MIN_SIZE = 600
MAX_SIZE = 1000

TRAIN_NUM_WORKS = 0
TEST_NUM_WORKS = 0

SHUFFLE = False
# batch size
BATCH_SIZE = 1

# VGG16
VGG16_LOAD_PATH = None
VGG16_USE_DROP = False
