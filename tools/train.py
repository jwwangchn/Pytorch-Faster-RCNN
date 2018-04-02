# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import sys
sys.path.append('../')

from libs.configs import cfgs
from torch.utils.data import DataLoader
from libs.dataload.voc import VOCDetection, VOC, Viz
from transforms import *


def data_loader():
    # transform = None
    voc = VOC()
    transform = Compose([Resize(min_size=cfgs.MIN_SIZE, max_size=cfgs.MAX_SIZE),  # Resize image and bboxes
                         Flip(x_random=True, y_random=False),  # Flip image and bboxes
                         Normalize(mean = voc.MEAN, std = voc.STD),
                         ToTensor(),
                         ])
    # Set data set
    trainset = VOCDetection(root=cfgs.VOC_DATA_ROOT,
                            image_set=[(2007, 'trainval')],
                            transforms = transform)

    testset = VOCDetection(root=cfgs.VOC_DATA_ROOT,
                           image_set=[(2007, 'test')],
                           transforms = transform)

    # Set dataloader
    trainloader = DataLoader(trainset,
                             batch_size=cfgs.BATCH_SIZE,
                             shuffle=cfgs.SHUFFLE,
                             num_workers=cfgs.TRAIN_NUM_WORKS)
    testloader = DataLoader(testset,
                            batch_size=cfgs.BATCH_SIZE,
                            shuffle=cfgs.SHUFFLE,
                            num_workers=cfgs.TEST_NUM_WORKS)

    return trainloader, testloader

def train():

    trainloader, testloader = data_loader()
    for img, bboxes, label in trainloader:
        print(img.shape)
        print(bboxes.shape)
        print(label.shape)
        break


if __name__ == '__main__':
    train()
