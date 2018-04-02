# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import torch.nn as nn
from libs.rpn import rpn_tools


class RegionProposalNetwork(nn.Module):
    def __init__(self,
                 in_channels = 512,
                 mid_channels = 512,
                 anchor_ratios = [0.5, 1,2 ],
                 anchor_scales = [8, 16, 32],
                 feat_stride = 16,
                 proposal_creator_params = dict()):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = rpn_tools.generate_anchor_base(anchor_scales=anchor_scales,
                                                          anchor_ratios=anchor_ratios)
        self.feat_base = feat_stride
        self.proposal_layer =