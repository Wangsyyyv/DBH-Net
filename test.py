from utils import *
import torch
from tqdm import tqdm
from metrics import calc_iou, eval_seg
from collections import OrderedDict
import argparse
import losses
import os
import torch.nn as nn
import torch.backends.cudnn as cudnn
from config import get_config
from unet import UNet
import torchvision

from datasets import LiverDataset
from torch.utils.data import DataLoader
import numpy as np
from STCUnet import STCUnet
import torch.nn.functional as F
import imageio
from PIL import Image
from utils_new import DiceLoss
import math

# 判断知否存在gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    # 模型名字
    parser.add_argument('--name', default=None,
                        help='model name: (default: model+timestamp)')
    # batch_size 大小
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--model', '-a', metavar='model', default='Swin_RFB')
    # 输入通道数
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    # 分类数
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')

    # loss 默认使用CrossEntropy 损失函数
    parser.add_argument('--loss', default='CrossEntropy')

    # dataset
    parser.add_argument('--dataset', default='liverdataset', help='dataset name')
    # 线程数 默认为0
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--cfg', type=str, default='configs/swin_tiny_patch4_window7_224_lite.yaml',
                        metavar="FILE", help='path to config file', )
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    config = parser.parse_args()
    config_new = get_config(config)

    return config, config_new


def validate(config, val_loader, model, criterion, save_path):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'mca': AverageMeter(),
                  'rec': AverageMeter(),
                  'pre': AverageMeter(),
                  'acc': AverageMeter(),
                  'speci': AverageMeter(),
                  'f1': AverageMeter(),
                  'miou': AverageMeter()}

    # switch to evaluate mode
    model.eval()
    dice_loss = DiceLoss(config['num_classes'])

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="val: ", ncols=100)
        for i, (input, target) in enumerate(pbar):
            input = input.to(device)
            target = target.to(device)
            output, o1, o2 = model(input)
            #output = model(input)

            loss_1 = criterion(o1, target.squeeze(1).long())
            loss_2 = criterion(o2, target.squeeze(1).long())
            loss_ce = criterion(output, target.squeeze(1).long())
            # loss_dice = dice_loss(output, target.squeeze(1), softmax=True)

            loss = 0.5 * loss_ce + 0.2 * loss_1 + 0.3 * loss_2
            #loss = loss_ce

            iou, rec, pre, acc, dice, speci, f1 = calc_iou(output, target, config['num_classes'])

            miou, mca = eval_seg(output, target, config['num_classes'])

            out = F.log_softmax(output, dim=1)
            pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
            gt_label = target.squeeze().cpu().data.numpy()

            pre_label = pre_label * 255
            gt_label = gt_label * 255

            for j in range(config['batch_size']):
                try:
                    img_pred = Image.fromarray(np.uint8(pre_label[j]))
                    # img_pred = img_pred.convert('L')
                    img_truth = Image.fromarray(np.uint8(gt_label[j]))
                    # img_truth = img_truth.convert('L')

                    img_pred.save(save_path + '/predict/' + 'pred_' + str(i) + '_' + str(j) + '.png')
                    img_truth.save(save_path + '/groundtruth/' + 'truth' + str(i) + '_' + str(j) + '.png')
                except IndexError:
                    print(j)

            # 平均损失
            avg_meters['loss'].update(loss.item(), input.size(0))
            # 平均IOU
            avg_meters['iou'].update(iou, input.size(0))
            # 平均Dice
            avg_meters['dice'].update(dice, input.size(0))
            # 平均分类准确度
            avg_meters['mca'].update(mca, input.size(0))
            # 精确度
            avg_meters['pre'].update(pre, input.size(0))
            # 召回率
            avg_meters['rec'].update(rec, input.size(0))
            # 准确度
            avg_meters['acc'].update(acc, input.size(0))

            avg_meters['miou'].update(miou, input.size(0))

            avg_meters['speci'].update(speci, input.size(0))

            avg_meters['f1'].update(f1, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('mca', avg_meters['mca'].avg),
                ('pre', avg_meters['pre'].avg),
                ('acc', avg_meters['acc'].avg),
                ('rec', avg_meters['rec'].avg),
                ('miou', avg_meters['miou'].avg),
                ('speci', avg_meters['speci'].avg),
                ('f1', avg_meters['f1'].avg)
            ])

            stdt = OrderedDict([
                ('loss_s', avg_meters['loss'].vari),
                ('iou_s', avg_meters['iou'].vari),
                ('dice_s', avg_meters['dice'].vari),
                ('mca_s', avg_meters['mca'].vari),
                ('pre_s', avg_meters['pre'].vari),
                ('acc_s', avg_meters['acc'].vari),
                ('rec_s', avg_meters['rec'].vari),
                ('miou_s', avg_meters['miou'].vari),
                ('speci_s', avg_meters['speci'].vari),
                ('f1_s', avg_meters['f1'].vari)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)

        pbar.close()

        return postfix, stdt


def main():
    print("====> testing...")
    config, config_new = parse_args()
    config = vars(config)
    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)
    # 损失函数
    if config['loss'] == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif config['loss'] == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = losses.__dict__[config['loss']]()
    cudnn.benchmark = True
    # create model
    print("=> creating model %s" % config['model'])
    save_path = './save_pred/'
    model = STCUnet(config_new, img_size=config['img_size'], num_classes=config['num_classes'])
    model.load_from(config_new)

    model_path = "./models/new_duomotai_no_CBAM_no_RFB_yes_res_300_TRAIN/model_294.pth"
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    val_root = "./data_new/test/images/"
    val_label = "./data_new/test/masks/"
    val_dataset = LiverDataset(val_root, val_label, train=False, transform=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True,
                                num_workers=config['num_workers'])

    val_log, std_log = validate(config, val_dataloader, model, criterion, save_path)

    print(
        'loss %.4f - iou %.4f - dice %.4f  - mean_class_accuracy %.4f - accuracy %.4f - recall %.4f - precision %.4f - miou %.4f - specificity %.4f - f1 %.4f'
        % (val_log['loss'], val_log['iou'], val_log['dice'], val_log['mca'], val_log['acc'], val_log['rec'],
           val_log['pre'], val_log['miou'], val_log['speci'], val_log['f1']))

    print('------------------------')
    print(
        'loss %.4f - iou %.4f - dice %.4f  - mean_class_accuracy %.4f - accuracy %.4f - recall %.4f - precision %.4f - miou %.4f - specificity %.4f - f1 %.4f'
        % (std_log['loss_s'], std_log['iou_s'], std_log['dice_s'], std_log['mca_s'], std_log['acc_s'], std_log['rec_s'],
           std_log['pre_s'], std_log['miou_s'], std_log['speci_s'], std_log['f1_s']))


if __name__ == '__main__':
    main()
