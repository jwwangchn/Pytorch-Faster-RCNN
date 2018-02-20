from voc_dataset import VOCBboxDataset
from util import resize_bbox, random_flip, flip_bbox
import cv2
import numpy as np
from config import opt
from torchvision import transforms as tvtsf
import torch

def pytorch_normalize(img):
    normalize = tvtsf.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img))
    return img.numpy()


def caffe_normalize(img):
    img = img[[2,1,0], :, :]    # BGR
    img = img * 255.0
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3,1,1)
    img = (img - mean).astype(np.float32, copy=True)
    return img

def prepare(img, min_size = 600, max_size = 1000):
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
    scale = 1.0
    scale = min_size / min(H, W)

    if scale * max(H, W) > max_size:
        scale = max_size / max(H, W)

    img = cv2.resize(img, (int(H * scale), int(H * scale)))
    img = img / 255.0
    if opt.caffe_pretrain:
        return caffe_normalize(img)
    else:
        return pytorch_normalize(img)


class Transform():
    def __init__(self, min_size = 600, max_size = 1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = prepare(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = resize_bbox(bbox, (H, W), (o_H, o_W))

        img, params = random_flip(img, x_random = True, y_random = False, return_param = True)
        bbox = flip_bbox(bbox, (o_H, o_W), x_flip = params['x_flip'], y_flip = params['y_flip'])

        return img, bbox, label, scale

class Dataset:
    def __init__(self, root = './data', train = True):
        if train:
            split = 'trainval'
        else:
            split = 'test'
        self.db = VOCBboxDataset(data_dir = root, split = split)
        self.tsf = Transform(opt.min_size, opt.max_size)


    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img, bbox, label, scale = self.tsf((ori_img, bbox, label))

        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)