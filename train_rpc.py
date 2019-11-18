import sys

import math

sys.path.append('.')
sys.path.append('..')
from lib.networks.model_repository import *
from lib.datasets.checkout import CheckoutDetection, CHECKOUT_ROOT
from lib.models.losses import FocalLoss
from lib.models.utils import _sigmoid
from lib.networks.net_utils import smooth_l1_loss, load_model, save_model, adjust_learning_rate
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
import utils
import os
import time
from collections import OrderedDict
import argparse

parser = argparse.ArgumentParser(
    description='Checkout Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset_root', default=CHECKOUT_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
# pvnet的学习率就是1e-3
parser.add_argument('--lr', '--learning-rate', default=1.25e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--num_epochs', default=140, type=int,
                    help='number of epochs for train')
parser.add_argument('--begin_epoch', default=0, type=int,
                    help='start epoch')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--save_interval', default=500, type=int,
                    help='Interval to save model weights')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train_one_epoch(model, optimizer, data_loader, device, epoch, args, print_freq):
    model.train()

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    cnt = 0
    for index, (images, targets) in enumerate(data_loader):
        cnt += 1
        images = images.to(device)
        targets = targets.to(device)
        predicts = model(images)
        # predicts = _sigmoid(predicts)
        # targets = targets.double()
        # predicts = predicts.double()
        # loss_value = loss(predicts, targets)    # mse
        loss_value = smooth_l1_loss(predicts, targets)

        if not math.isfinite(loss_value.item()):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_value)
            continue
            # sys.exit(1)

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        if index % print_freq == 0:
            print(f'进度:{index / len(data_loader) * 100}%, 损失值:{round(loss_value.item(), 7)}, '
                  f'输出最大值:{round(torch.max(predicts).item(), 3)}')
        if lr_scheduler is not None:
            lr_scheduler.step()  # 学习率调整一下

        if cnt % args.save_interval == 0:
            torch.save(model.state_dict(), 'weights/CountNet_epoch_%d_steps_%d_%f' % (epoch, cnt, loss_value.item()) + '.pth')
            print("save model to ", 'weights/CountNet_epoch_%d_steps_%d_%f' % (epoch, cnt, loss_value.item()) + '.pth')


def train_net():
    if os.name == 'nt':
        args.batch_size = 1
        print("running on my own xps13, so set batch_size to 1!")

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    model = Resnet18_8s(ver_dim=201)  # 这是训练一个类别的
    model.to(device)

    dataset = CheckoutDetection(CHECKOUT_ROOT, 'val')

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        model.load_state_dict(torch.load(args.resume, map_location=lambda storage, location: storage))
    else:

        print("train from scratch")
        # construct an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # loss = torch.nn.L1Loss()    # 换成focal loss
    # loss = FocalLoss()
    # loss = smooth_l1_loss()
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[90, 120],
                                                        gamma=0.1)

    # 数据加载
    # 数据增强后的dataset
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    begin_epoch = args.begin_epoch
    for epoch in range(begin_epoch, args.num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, args, print_freq=10)
        # update the learning rate
        lr_scheduler.step()

    print("That's it!")


if __name__ == "__main__":
    train_net()
