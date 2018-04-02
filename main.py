from torch.utils.data import DataLoader
from libs.dataload.voc import VOCDetection
from transforms import *

transform = Compose([Resize(min_size=600, max_size=1000),  # Resize image and bboxes
                     Flip(x_random=True, y_random=False),  # Flip image and bboxes
                     ])
# transform = Resize(min_size=600,max_size=1000)
# transform = Flip(x_random=True, y_random=False)
# transform = T.ToPILImage()
# print(transform)

dataset = VOCDetection(root='/home/ubuntu/data/VOCdevkit',
                       image_set=[(2007, 'test')],
                       transforms=transform)

trainloader = DataLoader(dataset,
                         batch_size=1,
                         shuffle=True,
                         num_workers=1)
# for img, bboxes, label in trainloader:
#     pass

for index in np.arange(len(dataset)):
    img, bboxes, label = dataset[index]
    # print(img.shape)
    if bboxes == []:
        print('failes')
    # print(label)

    # voc_Viz = Viz()
    # img = voc_Viz.draw_bbox(img, bboxes, label)
    #
    # print(img.shape)
    # print(bboxes)
    # print(label)
    # # img = img.transpose((1, 2, 0))  # HWC
    # cv2.imshow('preview', img[:, :, ::-1])
    # cv2.waitKey(5000)
