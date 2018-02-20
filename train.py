import torch
from dataset import Transform, Dataset
from config import opt
import fire
import cv2
from torch.utils.data import DataLoader
from torchvision import transforms as T
from voc import VOCDetection, VOC, Viz
from transforms import *


def data_loader(opt):
    # transform = None
    voc = VOC()
    transform = Compose([Resize(min_size=600, max_size=1000),  # Resize image and bboxes
                         Flip(x_random=True, y_random=False),  # Flip image and bboxes
                         Normalize(mean = voc.MEAN, std = voc.STD),
                         ToTensor(),
                         ])
    # Set data set
    trainset = VOCDetection(root='/home/ubuntu/data/VOCdevkit',
                            image_set=[(2007, 'trainval')],
                            transforms = transform)

    testset = VOCDetection(root='/home/ubuntu/data/VOCdevkit',
                           image_set=[(2007, 'test')],
                           transforms = transform)

    # Set dataloader
    trainloader = DataLoader(trainset,
                             batch_size=opt.batch_size,
                             shuffle=False,
                             num_workers=opt.train_num_works)
    testloader = DataLoader(testset,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=opt.test_num_works)

    return trainloader, testloader


def train(**kwargs):
    opt._parse(kwargs)
    voc_Viz = Viz()
    trainloader, testloader = data_loader(opt)

    for img, bboxes, label in testloader:

        print(img.shape)
        print(bboxes.shape)
        print(label.shape)
        break


if __name__ == '__main__':
    fire.Fire()
