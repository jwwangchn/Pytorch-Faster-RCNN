import torch
from torch import nn
from torchvision.models import vgg16
from config import opt
from torchvision.models import vgg16


class FasterRCNN(nn.Module):
    def __init__(self, extractor, rpn, head, mean,
                 min_size = 600, max_size = 1000,
                 loc_normalize_mean = (0.0, 0.0, 0.0, 0.0),
                 loc_normalize_std = (0.1, 0.1, 0.2, 0.2)):
        super(FasterRCNN, self).__init__()
        with self.init_scope():
            self.extractor = extractor
            self.rpn = rpn
            self.head = head

        self.mean = mean
        self.min_size = min_size
        self.max_size = max_size
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std