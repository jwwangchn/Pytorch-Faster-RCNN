# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import six

def generate_anchor_base(base_size = 16,
                         anchor_ratios = [0.5, 1, 2],
                         anchor_scales = [8, 16, 32]):
    """
    Generate anchor base windows by enumerating aspect ratio and scales
    :param base_size:
    :param anchor_ratios:
    :param anchor_scales:
    :return: An array of shape (R, 4), (y_min, x_min, y_max, x_max)
    """
    py = base_size / 2.0
    px = base_size / 2.0

    anchor_base = np.zeros((len(anchor_ratios) * len(anchor_scales), 4),
                           dtype=np.float32)
    for i in six.moves.range(len(anchor_ratios)):
        for j in six.moves.range(len(anchor_scales)):
            # 保证面积大小一定的情况下, 满足比例条件
            h = base_size * anchor_scales[j] * np.sqrt(anchor_ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / anchor_ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base

if __name__ == '__main__':
    print(generate_anchor_base())