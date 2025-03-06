from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd

from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import os
import sys
import time

import pytz
import wandb

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import warnings

warnings.filterwarnings("ignore")


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--method', default='dark', type=str,
                    help='Method for training')
parser.add_argument('--dataset_root', default='Exdark/',
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
args = parser.parse_args()


def train():
    KST = pytz.timezone('Asia/Seoul')

    method = args.method
    num_epochs = 100
    batch_size = 160
    num_workers = 8

    lr_init = 0.0001                   # initial learning rate (SGD=1E-2, Adam=1E-3)
    lr_min = 0.000001                   # minimum learning rate
    lr_max = 0.001
    warmup_ratio = 0.1

    resume = None
    use_wandb = False

    project_name = 'Exdark'
    exp_name = f'SSD-{method}'

    if not resume:
        save_dir = Path('runs')
        timestamp = datetime.now(KST).strftime('%Y%m%d_%H%M%S')
        save_dir = save_dir / f"{timestamp}_{exp_name}"
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / 'weights').mkdir(parents=True, exist_ok=True)
        (save_dir / 'pr_curve').mkdir(parents=True, exist_ok=True)
    else:
        save_dir = Path(resume).parent

    if use_wandb:
        wandb.init(project=project_name, name=exp_name)
        wandb.config.update({
            'timestamp': timestamp,
            'method': method,
            'save_dir': save_dir
        })

    cfg = voc
    dataset = VOCDetection(root=args.dataset_root,
                            transform=SSDAugmentation(cfg['min_dim'], MEANS),
                            method=method)

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        # cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        pass
        # vgg_weights = torch.load(args.save_folder + args.basenet)
        # print('Loading base network...')
        # ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    print('Loading the dataset...')

    data_loader = data.DataLoader(dataset, batch_size,
                                  num_workers=num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    num_steps_per_epoch = len(data_loader) # 데이터 사이즈? 7000
    total_steps = num_epochs * num_steps_per_epoch

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr_max,  # 최대 학습률
        total_steps=total_steps,
        pct_start=warmup_ratio,  # 전체 학습 단계 중 10%를 워밍업으로 사용
        anneal_strategy='cos',
        cycle_momentum=False  # AdamW에서는 모멘텀 사용 안 함
    )

    for epoch in range(num_epochs):
        tbar = tqdm(data_loader)
        for images, targets in tbar:
            images = images.cuda()
            targets = [ann.cuda() for ann in targets]

            optimizer.zero_grad()
            out = net(images)
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            scheduler.step()

            loc_loss += loss_l.item()
            conf_loss += loss_c.item()

            tbar.set_description(f'Epoch {epoch} | Loss: {loss.item():.4f} | Loc Loss: {loss_l.item():.4f} | Conf Loss: {loss_c.item():.4f}')

        if use_wandb:
            wandb.log({
                'train/loss': loss.item(),
                'train/loc_loss': loss_l.item(),
                'train/conf_loss': loss_c.item(),
                'train/lr': optimizer.param_groups[0]['lr'],
            }, step=epoch)

        torch.save({
            'model': ssd_net.state_dict(),
            'epoch': epoch,
            'loss': loss.item(),
        }, save_dir / 'weights' / f'epoch_{epoch}.pth')

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

if __name__ == '__main__':
    train()
