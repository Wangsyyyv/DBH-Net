from torch.nn import init
import os
import shutil
import random


### initalize the module
def init_weights(net, init_type='normal'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


### compute model params
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


import argparse


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vari = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vari += (self.val - self.avg) ** 2


def moveFile(src_path, dest_path, index_list):
    files = os.listdir(src_path)
    for i in index_list:
        print(files[i])
        file_path = src_path + '/' + files[i]
        new_file_path = dest_path + '/' + files[i]
        shutil.move(file_path, new_file_path)

if __name__ == '__main__':
    random_num = random.sample(range(20, 400), 80)
    image_path = './data/train'
    image_new_path = './data/val'

    label_path = './data/train_label'
    label_new_path = './data/val_label'

    moveFile(image_path,image_new_path,random_num)
    moveFile(label_path,label_new_path,random_num)

