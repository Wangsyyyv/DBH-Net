import argparse
import os
from collections import OrderedDict
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm
from datasets import LiverDataset
from torch.utils.data import DataLoader

from utils import AverageMeter, str2bool
import losses
from metrics import calc_iou
import torch.nn.functional as F
import numpy as np
from STCUnet import STCUnet
from config import get_config
from utils_new import DiceLoss
from unet import UNet
from thop import profile

LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')
LOSS_NAMES.append('CrossEntropyLoss')

# 判断知否存在gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 设置参数
def parse_args():
    parser = argparse.ArgumentParser()
    # 模型名字
    parser.add_argument('--name', default=None,
                        help='model name: (default: model+timestamp)')
    # epoch 选择
    parser.add_argument('--epochs', default=250, type=int, metavar='N',
                        help='number of total epochs to run')
    # batch_size 大小
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--model', '-a', metavar='model', default='kvasir_SwinUnet')
    # 输入通道数
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    # 分类数
    parser.add_argument('--num_classes', default=2
                        , type=int,
                        help='number of classes')

    # loss 默认使用CrossEntropy 损失函数
    parser.add_argument('--loss', default='CrossEntropy', choices=LOSS_NAMES,
                        help='loss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: BCEDiceLoss)' +
                             'binary_cross_entropy_with_logits')

    # dataset
    parser.add_argument('--dataset', default='gastric', help='dataset name')
    # 线程数 默认为0
    parser.add_argument('--num_workers', default=0, type=int)

    # optimizer 默认Adam
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool, help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='ReduceLROnPlateau',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    # 最小学习率
    parser.add_argument('--min_lr', default=1e-5, type=float, help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
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

    # 提前停止
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')

    config = parser.parse_args()
    config_new = get_config(config)

    return config, config_new


# 训练函数
def train(config, train_loader, model, criterion, optimizer, epoch=1):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter(), 'dice': AverageMeter()}

    model.train()

    pbar = tqdm(train_loader, desc="train: ", ncols=100)

    for input, target in pbar:
        input = input.to(device)
        target = target.to(device)
        output, o1, o2 = model(input)

        loss_1 = criterion(o1, target.squeeze(1).long())
        loss_2 = criterion(o2, target.squeeze(1).long())
        loss_ce = criterion(output, target.squeeze(1).long())

        loss = 0.5 * loss_ce + 0.2 * loss_1 + 0.3 * loss_2

        iou = calc_iou(output, target, config['num_classes'])[0]

        dice = 2 - 2 / (iou + 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 平均损失
        avg_meters['loss'].update(loss.item(), input.size(0))
        # 平均IOU
        avg_meters['iou'].update(iou, input.size(0))
        # 平均Dice
        avg_meters['dice'].update(dice, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', avg_meters['dice'].avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])


# 验证
def validate(config, val_loader, model, criterion):
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
        for input, target in pbar:
            input = input.to(device)
            target = target.to(device)

            output, o1, o2 = model(input)

            loss_1 = criterion(o1, target.squeeze(1).long())
            loss_2 = criterion(o2, target.squeeze(1).long())
            loss_ce = criterion(output, target.squeeze(1).long())
            # loss_dice = dice_loss(output, target.squeeze(1), softmax=True)

            loss = 0.5 * loss_ce + 0.2 * loss_1 + 0.3 * loss_2

            iou, rec, pre, acc, dice, speci, f1 = calc_iou(output, target, config['num_classes'])
            # dice = 2 - 2 / (iou + 1)

            # 平均损失
            avg_meters['loss'].update(loss.item(), input.size(0))
            # 平均IOU
            avg_meters['iou'].update(iou, input.size(0))
            # 平均Dice
            avg_meters['dice'].update(np.mean(dice), input.size(0))
            avg_meters['pre'].update(pre, input.size(0))
            avg_meters['rec'].update(rec, input.size(0))
            avg_meters['acc'].update(acc, input.size(0))
            avg_meters['speci'].update(speci, input.size(0))
            avg_meters['f1'].update(f1, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('pre', avg_meters['pre'].avg),
                ('acc', avg_meters['acc'].avg),
                ('rec', avg_meters['rec'].avg),
                ('speci', avg_meters['speci'].avg),
                ('f1', avg_meters['f1'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

        return OrderedDict([('loss', avg_meters['loss'].avg),
                            ('iou', avg_meters['iou'].avg),
                            ('dice', avg_meters['dice'].avg),
                            ('pre', avg_meters['pre'].avg),
                            ('acc', avg_meters['acc'].avg),
                            ('rec', avg_meters['rec'].avg),
                            ('speci', avg_meters['speci'].avg),
                            ('f1', avg_meters['f1'].avg)
                            ])


def main():
    config, config_new = parse_args()
    config = vars(config)

    if config['name'] is None:
        config['name'] = '%s_%s_TRAIN' % (config['dataset'], config['model'])
    os.makedirs('./models/%s' % config['name'], exist_ok=True)


    # 将配置参数写入到文件中
    with open('./models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # 损失函数
    if config['loss'] == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif config['loss'] == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = losses.__dict__[config['loss']]()

    cudnn.benchmark = True
    # create model
    print("-----model %s-----" % config['model'])
    model = STCUnet(config_new, img_size=config['img_size'], num_classes=config['num_classes'])
    model.load_from(config_new)
    model = model.to(device)

    dummy_input = torch.randn(8, 3, 224, 224)
    flops, params = profile(model, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # 获取数据
    train_root = "./data_gastric/train/images/"
    train_label = "./data_gastric/train/masks/"
    liver_dataset = LiverDataset(train_root, train_label, train=True, transform=True)
    dataloaders = DataLoader(liver_dataset, batch_size=config['batch_size'], shuffle=True,
                             num_workers=config['num_workers'])

    val_root = "./data_gastric/val/images/"
    val_label = "./data_gastric/val/masks/"
    val_dataset = LiverDataset(val_root, val_label, train=False, transform=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True,
                                num_workers=config['num_workers'])
    # 日志
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
        ('val_pre', []),
        ('val_acc', []),
        ('val_rec', []),
        ('val_speci', []),
        ('val_f1', [])
    ])

    best_iou = 0
    trigger = 0

    for epoch in range(config['epochs']):
        print('\nEpoch [%d/%d]' % (epoch, config['epochs']))

        train_log = train(config, dataloaders, model, criterion, optimizer)
        val_log = validate(config, val_dataloader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('\nloss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        log['val_pre'].append(val_log['pre'])
        log['val_acc'].append(val_log['acc'])
        log['val_rec'].append(val_log['rec'])
        log['val_speci'].append(val_log['speci'])
        log['val_f1'].append(val_log['f1'])

        # 将日志写入文件中
        pd.DataFrame(log).to_csv('./models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), './models/%s/model_%d.pth' %
                       (config['name'], epoch))
            best_iou = val_log['iou']
            print("-----saved model-----")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
