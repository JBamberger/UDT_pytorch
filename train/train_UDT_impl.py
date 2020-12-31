import os
import time
from os.path import isfile

import numpy as np
import torch
from torch import nn as nn
import torch.fft as fft
from torch.backends import cudnn as cudnn

import config
import util
from train.DCFNet import DCFNet
from util import AverageMeter, output_drop, create_fake_y


class TrackerConfig(object):
    crop_sz = 125
    output_sz = 121
    lambda0 = 1e-4
    padding = 2.0
    output_sigma_factor = 0.1
    output_sigma = crop_sz / (1 + padding) * output_sigma_factor

    # shape: 121, 121
    y = torch.tensor(util.gaussian_shaped_labels(output_sigma, [output_sz, output_sz]).astype(np.float32)).cuda()

    # shape: 1, 1, 121, 61, 2
    yf = fft.rfftn(y.view(1, 1, output_sz, output_sz), dim=[-2, -1])

    # cos_window = torch.Tensor(np.outer(np.hanning(crop_sz), np.hanning(crop_sz))).cuda()  # train without cos window


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    initial_y = tracker_config.y.clone()
    label = tracker_config.yf.repeat(args.batch_size * gpu_num, 1, 1, 1)

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

    initial_y = tracker_config.y.clone()
    # Shape: batch, 1, 121, 61
    label = tracker_config.yf.repeat(args.batch_size * gpu_num, 1, 1, 1)

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
    # compute the model response for the template, search region and label
    with torch.no_grad():
        response = model(template, search, label)

    fake_y = create_fake_y(initial_y, response)
    return fft.rfftn(fake_y, dim=[-2, -1]).cuda(non_blocking=True)


def main(pargs, pgpu_num, ptrain_loader, pval_loader):
    global tracker_config, target, args, gpu_num, train_loader, val_loader
    args = pargs
    gpu_num = pgpu_num
    train_loader = ptrain_loader
    val_loader = pval_loader

    best_loss = 1e6
    tracker_config = TrackerConfig()
    model = DCFNet(config=tracker_config).cuda()

    print('GPU NUM: {:2d}'.format(gpu_num))
    if gpu_num > 1:
        model = torch.nn.DataParallel(model, list(range(gpu_num))).cuda()
    criterion = nn.MSELoss(reduction='sum').cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=util.compute_lr_gamma(args.lr, 1e-5, args.epochs))

    # for training
    target = tracker_config.y.unsqueeze(0).unsqueeze(0).repeat(args.batch_size * gpu_num, 1, 1, 1)

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
