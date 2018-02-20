from pprint import pprint
class Config:

    # data
    voc_data_dir = '/home/ubuntu/data/VOCdevkit/VOC2007'
    min_size = 600
    max_size = 1000
    train_num_works = 0
    test_num_works = 0
    batch_size = 1

    # sigma for l1_smooth_loss
    rpn_sigma = 3.0
    roi_sigma = 1.0

    # param for optimizer
    weight_decay = 0.0005
    lr_decay = 0.1
    lr = 1e-3

    # preset
    data = 'voc'
    pretrained_model = 'vgg16'

    # training
    epoch = 14

    use_adam = False
    use_drop = False

    # debug
    debug_file = '/tmp/debugf'

    test_num = 10000

    # model
    load_path = None

    caffe_pretrain = False
    caffe_pretrain_path = 'checkpoints/vgg16-caffe.path'

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

opt = Config()