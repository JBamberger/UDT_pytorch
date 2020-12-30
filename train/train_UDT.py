import argparse
import os
import time
from os.path import isfile

import numpy as np
import torch
import torch.fft as fft
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import dataloader

import config
import util
from train.DCFNet import DCFNet
from dataset import ILSVRC2015
from util import AverageMeter, output_drop


class TrackerConfig(object):
    crop_sz = 125
    output_sz = 121
    lambda0 = 1e-4
    padding = 2.0
    output_sigma_factor = 0.1
    output_sigma = crop_sz / (1 + padding) * output_sigma_factor

    # shape: 121, 121
    y = util.gaussian_shaped_labels(output_sigma, [output_sz, output_sz]).astype(np.float32)

    # shape: 1, 1, 121, 61, 2
    yf = fft.rfftn(torch.Tensor(y).view(1, 1, output_sz, output_sz).cuda(), dim=[-2, -1])

    # cos_window = torch.Tensor(np.outer(np.hanning(crop_sz), np.hanning(crop_sz))).cuda()  # train without cos window


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    initial_y = tracker_config.y.copy()
    label = tracker_config.yf.repeat(args.batch_size * gpu_num, 1, 1, 1).cuda(non_blocking=True)

    end = time.time()
    for i, (template, search1, search2) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        output = do_tracking(initial_y, label, model, search1, search2, template)
        # consistency loss. target is the initial Gaussian label
        loss = criterion(output, target) / (args.batch_size * gpu_num)
        # measure accuracy and record loss
        losses.update(loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:



            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:2.3f} ({batch_time.avg:2.3f})\t'
                  f'Data {data_time.val:2.3f} ({data_time.avg:2.3f})\t'
                  f'Loss {losses.val:2.4f} ({losses.avg:2.4f})\t')


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.eval()

    initial_y = tracker_config.y.copy()
    # Shape: batch, 1, 121, 61
    label = tracker_config.yf.repeat(args.batch_size * gpu_num, 1, 1, 1).cuda(non_blocking=True)

    with torch.no_grad():
        end = time.time()
        for i, (template, search1, search2) in enumerate(val_loader):

            output = do_tracking(initial_y, label, model, search1, search2, template)
            loss = criterion(output, target) / (args.batch_size * gpu_num)

            # measure accuracy and record loss
            losses.update(loss.item())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses))

        print(' * Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses))

    return losses.avg


def do_tracking(initial_y, label, model, search1, search2, template):
    # template, search1, search2 shape: batch_size, channels, height, width

    template = template.cuda(non_blocking=True)
    search1 = search1.cuda(non_blocking=True)
    search2 = search2.cuda(non_blocking=True)
    if gpu_num > 1:
        template_feat = model.module.feature(template)
        search1_feat = model.module.feature(search1)
        search2_feat = model.module.feature(search2)
    else:
        template_feat = model.feature(template)
        search1_feat = model.feature(search1)
        search2_feat = model.feature(search2)

    # forward tracking 1
    fake_yf = forward_tracking(model, template_feat, search1_feat, label, initial_y)

    # forward tracking 2
    fake_yf = forward_tracking(model, search1_feat, search2_feat, fake_yf, initial_y)

    # backward tracking
    output = model(search2_feat, template_feat, fake_yf)

    # the sample dropout is necessary, otherwise we find the loss tends to become unstable
    output = output_drop(output, target)

    return output


def forward_tracking(model, template, search, label, initial_y):
    batch_size = args.batch_size * gpu_num

    # compute the model response for the template, search region and label
    with torch.no_grad():
        response = model(template, search, label)

    return create_fake_y(batch_size, initial_y, response)


def create_fake_y(batch_size, initial_y, response):
    # find the peak location indices in the flattened response maps
    peak, index = torch.max(response.view(batch_size, -1), dim=1)

    # convert the indices in the flattened response map to 2D coordinates in the plane
    r_max, c_max = np.unravel_index(index.cpu(), [tracker_config.output_sz, tracker_config.output_sz])

    fake_y = np.zeros((batch_size, 1, tracker_config.output_sz, tracker_config.output_sz))
    for j in range(batch_size):
        shift_y = np.roll(initial_y, r_max[j])
        fake_y[j, ...] = np.roll(shift_y, c_max[j])

    # Compute fourier transform of the label
    fake_yview = torch.Tensor(fake_y) \
        .view(batch_size, 1, tracker_config.output_sz, tracker_config.output_sz) \
        .cuda()
    return fft.rfftn(fake_yview, dim=[-2, -1]).cuda(non_blocking=True)


def main():
    global args, tracker_config, gpu_num, target, args
    parser = argparse.ArgumentParser(description='Training DCFNet in PyTorch')
    parser.add_argument('--input_sz', dest='input_sz', default=125, type=int, help='crop input size')
    parser.add_argument('--padding', dest='padding', default=2.0, type=float, help='crop padding size')
    parser.add_argument('--range', dest='range', default=10, type=int, help='select range')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-5, type=float,
                        metavar='W', help='weight decay (default: 5e-5)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--save', '-s', default=None, type=str, help='directory for saving')
    args = parser.parse_args()
    print(args)
    best_loss = 1e6
    tracker_config = TrackerConfig()
    model = DCFNet(config=tracker_config).cuda()
    gpu_num = torch.cuda.device_count()
    print('GPU NUM: {:2d}'.format(gpu_num))
    if gpu_num > 1:
        model = torch.nn.DataParallel(model, list(range(gpu_num))).cuda()
    criterion = nn.MSELoss(reduction='sum').cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=util.compute_lr_gamma(args.lr, 1e-5, args.epochs))
    # for training
    target = torch.Tensor(tracker_config.y) \
        .cuda() \
        .unsqueeze(0) \
        .unsqueeze(0) \
        .repeat(args.batch_size * gpu_num, 1, 1, 1)
    # optionally resume from a checkpoint
    if args.resume:
        if isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True
    # training data
    # crop_base_path = os.path.join(config.dataset_root, 'ILSVRC2015', f'crop_{args.input_sz:d}_{args.padding:1.1f}')
    # if not isdir(crop_base_path):
    #     print(f'please run gen_training_data.py --output_size {args.input_sz:d} --padding {args.padding:.1f}!')
    #     exit()
    checkpoint_path = args.save if args.save else config.checkpoint_root
    checkpoint_saver = util.CheckpointSaver(
        os.path.join(checkpoint_path, f'crop_{args.input_sz:d}_{args.padding:1.1f}'))
    # Create training dataset loader
    train_dataset = ILSVRC2015(train=True, range=args.range)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size * gpu_num, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    # Create validation dataset loader
    val_dataset = ILSVRC2015(train=False, range=args.range)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size * gpu_num, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    # Bring the lr scheduler to the first epoch
    for epoch in range(args.start_epoch):
        lr_scheduler.step()
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        loss = validate(val_loader, model, criterion)

        # remember best loss and save checkpoint
        is_best = loss < best_loss
        best_loss = min(best_loss, loss)
        checkpoint_saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best)

        # Update the learning rate
        lr_scheduler.step()


if __name__ == '__main__':
    main()
