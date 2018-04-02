# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np

import torch.nn as nn

class FasterRCNN(nn.Module):
    def __init__(self, extractor, rpn, head):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head

    @property
    def n_class(self):
        return self.head.n_class

    def forward(self, x, scale = 1.0):
        img_size = x.shape[2:]

        # 特征提取层
        h = self.extractor(x)
        # RPN 层
        rpn_loss, rpn_score, rois, roi_indices, anchor = self.rpn(h,
                                                                  img_size,
                                                                  scale)
        # 分类层
        roi_cls_locs, roi_scores = self.head(h,
                                             rois,
                                             roi_indices)
        return roi_cls_locs, roi_scores, rois, roi_indices

