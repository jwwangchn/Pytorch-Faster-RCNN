# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import


class ProposalCreator:
    def __init__(self,
                 parent_model,
                 nms_thread=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16):
        self.parent_model = parent_model
        self.nms_thread = nms_thread
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_test_pre_nms
        self.min_size = min_size

    def __call__(self,
                 loc,
                 score,
                 anchor,
                 img_size,
                 scale = 1.0):

