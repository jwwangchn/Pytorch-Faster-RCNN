# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

from faster_rcnn import FasterRCNN
from libs.vgg import vgg16


class FasterRCNNVGG16(FasterRCNN):
    def __init__(self,
                 n_fg_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]):
        extractor, classifier = vgg16.decom_vgg16()
        super(FasterRCNNVGG16, self).__init__(
            extractor,
        )

