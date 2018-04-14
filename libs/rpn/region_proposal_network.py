# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import torch.nn as nn
from libs.rpn import rpn_tools


class ProposalCreator:
    """
    Proposal Layer
    """
    def __init__(self,
                 parent_model,
                 nms_thresh = 0.7,
                 n_train_pre_nms = 12000,
                 n_train_post_nms = 2000,
                 n_test_pre_nms = 6000,
                 n_test_post_nms = 300,
                 min_size = 16):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale = 1.0):
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms


class AnchorTargetCreator():
    """
    Anchor Target Layer
    """
    pass

class ProposalTargetCreator():
    """
    Proposal Target Layer
    """
    pass

class RegionProposalNetwork(nn.Module):
    def __init__(self,
                 in_channels = 512,
                 mid_channels = 512,
                 anchor_ratios = [0.5, 1, 2],
                 anchor_scales = [8, 16, 32],
                 feat_stride = 16,
                 proposal_creator_params = dict()):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = rpn_tools.generate_anchor_base(anchor_scales=anchor_scales,
                                                          anchor_ratios=anchor_ratios)
        self.feat_base = feat_stride