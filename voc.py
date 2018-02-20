import numpy as np
import os
import torch.utils.data as data
from torchvision import transforms as T
import xml.etree.ElementTree as ET
import cv2
from PIL import Image

class VOC():
    CLASSES = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
    NUM_CLASSES = len(CLASSES)

    # Caffe
    # MEAN = [123.68, 116.779, 103.939]  # R,G,B

    # Pytorch
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    label_to_id = dict(map(reversed, enumerate(CLASSES)))
    id_to_label = dict(enumerate(CLASSES))

class Viz():
    def __init__(self):
        voc = VOC()
        self.classes = voc.CLASSES
        self.num_classes = voc.NUM_CLASSES
        self.label_to_id = voc.label_to_id
        self.id_to_label = voc.id_to_label

        colors = {}
        for label in self.classes:
            id = self.label_to_id[label]
            color = self._to_color(id, self.num_classes)
            colors[id] = color
            colors[label] = color
        self.colors = colors

    def _to_color(self, index, n_classes):
        base = int(np.ceil(pow(n_classes, 1. / 3)))
        base2 = base * base
        b = 2 - index / base2
        r = 2 - (index % base2) / base
        g = 2 - (index % base2) % base
        # return (b * 127, r * 127, g * 127)
        return (r * 127, g * 127, b * 127)

    def draw_bbox(self, img, bboxes, labels, relative = False):
        img = img.transpose((1, 2, 0))  # HWC
        if len(labels) == 0:
            return img
        img = img.copy()
        H, W, _ = img.shape

        if relative:
            bboxes = bboxes * [W, H, W, H]

        bboxes = bboxes.astype(np.int)
        labels = labels.astype(np.int)

        for bbox, label in zip(bboxes, labels):
            xmin, ymin, xmax, ymax = bbox
            color = self.colors[label]
            label = self.id_to_label[label]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 3)
            cv2.putText(img, label, (xmin + 1, ymin - 5), cv2.FONT_HERSHEY_DUPLEX, 0.4, color, 1, cv2.LINE_AA)

        return img


class ParseAnnotation():
    def __init__(self):
        voc = VOC()
        self.label_to_id = voc.label_to_id

    def __call__(self, target):
        tree = ET.parse(target)
        bboxes = []
        labels = []
        for obj in tree.findall('object'):
            label = obj.find('name').text.lower().strip()
            label = self.label_to_id[label]

            bndbox = obj.find('bndbox')
            bbox = [int(bndbox.find(_).text) - 1 for _ in ('xmin', 'ymin', 'xmax', 'ymax')]
            bboxes.append(bbox)
            labels.append(label)

        return np.array(bboxes), np.array(labels)



class VOCDetection(data.Dataset):
    def __init__(self, root, image_set, transforms=None, target_transforms=None):
        self.root = root
        self.image_set = image_set
        self.transforms = transforms
        self.target_transforms = target_transforms

        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self.parse_annotation = ParseAnnotation()

        # dataset information
        voc = VOC()
        self.classes = voc.CLASSES
        self.num_classes = voc.NUM_CLASSES
        self.label_to_id = voc.label_to_id
        self.id_to_label = voc.id_to_label

        self.ids = []
        for year, split in image_set:
            basepath = os.path.join(self.root, 'VOC' + str(year))
            path = os.path.join(basepath, 'ImageSets', 'Main')

            for file in os.listdir(path):
                if not file.endswith('_' + split + '.txt'):
                    continue
                with open(os.path.join(path, file)) as f:
                    for line in f:
                        self.ids.append((basepath, line.strip()[:-3]))

        self.ids = sorted(list(set(self.ids)), key=lambda _: _[0] + _[1])

    def __getitem__(self, index):
        img_id = self.ids[index]

        img = cv2.imread(self._imgpath % img_id)[:, :, ::-1]    # BGR -> RGB
        img = np.transpose(img, (2, 0, 1))  # C H W
        # Use PIL to read images, so we can use torchvision transforms tools
        # img = Image.open(self._imgpath % img_id)    # RGB

        bboxes, labels = self.parse_annotation(self._annopath % img_id)

        if self.transforms is not None:
            img, bboxes = self.transforms(img, bboxes)

        bboxes, labels = self.filter(img, bboxes, labels)
        if self.target_transforms is not None:
            bboxes, labels = self.target_transforms(bboxes, labels)

        return img, bboxes, labels

    def __len__(self):
        return len(self.ids)

    def filter(self, img, boxes, labels):
        shape = img.shape
        if len(shape) == 2:
            h, w = shape
        else:  # !!
            if shape[0] > shape[2]:  # HWC
                h, w = img.shape[:2]
            else:  # CHW
                h, w = img.shape[1:]

        boxes_ = []
        labels_ = []
        for box, label in zip(boxes, labels):
            if min(box[2] - box[0], box[3] - box[1]) <= 0:
                continue
            if np.max(boxes) < 1 and np.sqrt((box[2] - box[0]) * w * (box[3] - box[1]) * h) < 8:
                # if np.max(boxes) < 1 and min((box[2] - box[0]) * w, (box[3] - box[1]) * h) < 5:
                continue
            boxes_.append(box)
            labels_.append(label)
        return np.array(boxes_), np.array(labels_)

