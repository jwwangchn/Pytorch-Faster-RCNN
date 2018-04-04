# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
from libs.configs import cfgs

from torch import nn
from torchvision.models import vgg16

def decom_vgg16():
    model = vgg16(not cfgs.VGG16_LOAD_PATH)
    features = list(model.features)[:30]    # VGG16 feature 模型中 conv + relu + pool 一共有 31 层
    classifier = model.classifier
    classifier = list(classifier)
    del classifier[6]       # 删除掉了最后一个全连接层

    # 是否使用 dropout 层
    if not cfgs.VGG16_USE_DROP:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # 禁掉前几层的反向传播 (4个卷积层)
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier



