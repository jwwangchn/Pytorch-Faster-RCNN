import numpy as np
import cv2
import torch
from torchvision import transforms as T
import random


def resize_img(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray:
        A preprocessed image.

    """
    _, H, W = img.shape
    img = img.transpose((1, 2, 0))  # HWC
    scale = 1.

    scale = min_size / min(H, W)

    if scale * max(H, W) > max_size:
        scale = max_size / max(H, W)
    img = cv2.resize(img, (int(H * scale), int(W * scale)))
    img = img.transpose((2, 0, 1))  # CHW

    return img


def resize_bbox(bbox, in_size, out_size):
    """
    Resize bounding boxes according to image resize.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        in_size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        out_size (tuple): A tuple of length 2. The height and the width
            of the image after resized.

    Returns:
        ~numpy.ndarray:
        Bounding boxes rescaled according to the given image shapes.

    """
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]

    bbox[:, 0] = x_scale * bbox[:, 0]
    bbox[:, 2] = x_scale * bbox[:, 2]
    bbox[:, 1] = y_scale * bbox[:, 1]
    bbox[:, 3] = y_scale * bbox[:, 3]

    return bbox


class Resize():
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img, bboxes):
        _, H, W = img.shape
        img = resize_img(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bboxes = resize_bbox(bboxes, (H, W), (o_H, o_W))

        return img, bboxes


def random_flip(img,
                x_random=False,
                y_random=False,
                return_param=False,
                copy=False):
    """Randomly flip an image in vertical or horizontal direction.

    Args:
        img (~numpy.ndarray): An array that gets flipped. This is in
            CHW format.
        y_random (bool): Randomly flip in vertical direction.
        x_random (bool): Randomly flip in horizontal direction.
        return_param (bool): Returns information of flip.
        copy (bool): If False, a view of :obj:`img` will be returned.

    Returns:
        ~numpy.ndarray or (~numpy.ndarray, dict):

        If :obj:`return_param = False`,
        returns an array :obj:`out_img` that is the result of flipping.

        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`out_img, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.

        * **y_flip** (*bool*): Whether the image was flipped in the\
            vertical direction or not.
        * **x_flip** (*bool*): Whether the image was flipped in the\
            horizontal direction or not.

    """
    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        img = img[:, ::-1, :]
    if x_flip:
        img = img[:, :, ::-1]

    if copy:
        img = img.copy()

    if return_param:
        return img, {'y_flip': y_flip, 'x_flip': x_flip}
    else:
        return img


def flip_bbox(bbox,
              size,
              y_flip=False,
              x_flip=False):
    """Flip bounding boxes accordingly.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    Args:
        bbox (~numpy.ndarray): An array whose shape is :math:`(R, 4)`.
            :math:`R` is the number of bounding boxes.
        size (tuple): A tuple of length 2. The height and the width
            of the image before resized.
        y_flip (bool): Flip bounding box according to a vertical flip of
            an image.
        x_flip (bool): Flip bounding box according to a horizontal flip of
            an image.

    Returns:
        ~numpy.ndarray:
        Bounding boxes flipped according to the given flips.

    """
    H, W = size
    bbox = bbox.copy()
    if y_flip:
        y_max = H - bbox[:, 1]
        y_min = H - bbox[:, 3]
        bbox[:, 1] = y_min
        bbox[:, 3] = y_max

    if x_flip:
        x_max = W - bbox[:, 0]
        x_min = W - bbox[:, 2]
        bbox[:, 0] = x_min
        bbox[:, 2] = x_max
    return bbox


class Flip():
    def __init__(self, x_random, y_random):
        self.x_random = x_random
        self.y_random = y_random

    def __call__(self, img, bboxes):
        _, H, W = img.shape
        img, params = random_flip(img, x_random=self.x_random,
                                  y_random=self.y_random,
                                  return_param=True)
        bboxes = flip_bbox(bboxes, (H, W),
                           x_flip=params['x_flip'],
                           y_flip=params['y_flip'])

        return img, bboxes


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> T.Compose([
        >>>     T.CenterCrop(10),
        >>>     T.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img, bboxes)
        return img, bboxes


class ToTensor():
    def __init__(self):
        pass

    def __call__(self, img, bboxes):
        if isinstance(img, np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(img)).float(), bboxes


class Normalize():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, bboxes):
        img = img / 255.0
        normalize = T.Normalize(mean=self.mean, std=self.std)
        img = normalize(torch.from_numpy(img))
        return img.numpy(), bboxes
