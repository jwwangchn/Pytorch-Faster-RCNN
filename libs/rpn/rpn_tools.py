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

def bbox2loc(src_bbox, dst_bbox):
    """

    :param src_bbox: p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax} (Prediction)
    :param dst_bbox: g_{ymin}, g_{xmin}, g_{ymax}, g_{xmax} (Ground Truth)
    :return:
    """
    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + height / 2.0
    ctr_x = src_bbox[:, 1] + width / 2.0

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + base_height / 2.0
    base_ctr_x = dst_bbox[:, 3] + width / 2.0

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    loc = np.vstack((dy, dx, dh, dw)).transpose()
    return loc

def loc2bbox(src_bbox, loc):
    """

    :param src_bbox: p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}
    :param loc: t_y, t_x, t_h, t_w
    :return:
    """
    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype = loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy = False)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + src_height / 2.0
    src_ctr_x = src_bbox[:, 1] + src_width / 2.0

    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]

    dst_bbox = np.zeros(loc.shape, dtype = loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox


if __name__ == '__main__':
    print(generate_anchor_base())