import os
import time
from os.path import isfile

import numpy as np
import torch
import torch.fft as fft
from torch import nn as nn
from torch.backends import cudnn as cudnn

import config
import util
from train.DCFNet import DCFNet
from util import AverageMeter, output_drop, create_fake_y


class UdtTrainer(object):

    def __init__(self,
                 args, gpu_num: int, train_loader, val_loader,
                 crop_sz=125,
                 output_sz=121,
                 lambda0=1e-4,
                 padding=2.0,
                 output_sigma_factor=0.1):
        self.crop_sz = crop_sz
        self.output_sz = output_sz
        self.lambda0 = lambda0
        self.padding = padding
        output_sigma = crop_sz / (1 + padding) * output_sigma_factor

        self.args = args
        self.gpu_num = gpu_num
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = args.batch_size * gpu_num

        self.best_loss = 1e6

        # shape: 121, 121
        self.y = torch.tensor(util.gaussian_shaped_labels(output_sigma, [self.output_sz, self.output_sz])
                              .astype(np.float32)).cuda()
        # shape: 1, 1, 121, 61, 2
        self.yf = fft.rfftn(self.y.view(1, 1, self.output_sz, self.output_sz), dim=[-2, -1])
        # Shape: 121, 121
        self.initial_y = self.y.clone()
        # Shape: batch, 1, 121, 61
        self.label = self.yf.repeat(self.batch_size, 1, 1, 1)

        self.model = DCFNet(lambda0=self.lambda0).cuda()

        print('GPU NUM: {:2d}'.format(gpu_num))
        if gpu_num > 1:
            self.model = torch.nn.DataParallel(self.model, list(range(gpu_num))).cuda()

        self.criterion = nn.MSELoss(reduction='sum').cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr,
                                         momentum=args.momentum, weight_decay=args.weight_decay)

        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=util.compute_lr_gamma(args.lr, 1e-5, args.epochs))

        # Bring the lr scheduler to the first epoch
        for epoch in range(args.start_epoch):
            self.lr_scheduler.step()

        # for training
        self.target = self.y.unsqueeze(0).unsqueeze(0).repeat(args.batch_size * gpu_num, 1, 1, 1)

        # optionally resume from a checkpoint
        if args.resume:
            if isfile(args.resume):
                print(f"=> loading checkpoint '{args.resume}'")
                checkpoint = torch.load(args.resume)
                self.args.start_epoch = checkpoint['epoch']
                self.best_loss = checkpoint['best_loss']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            else:
                print(f"=> no checkpoint found at '{args.resume}'")
        cudnn.benchmark = True

        checkpoint_path = args.save if args.save else config.checkpoint_root
        self.checkpoint_saver = util.CheckpointSaver(
            save_path=os.path.join(checkpoint_path, f'crop_{args.input_sz:d}_{args.padding:1.1f}'),
            verbose=True)

    def __call__(self):
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.train(epoch)
            loss = self.validate()

            # remember best loss and save checkpoint
            is_best = loss < self.best_loss
            self.best_loss = min(self.best_loss, loss)
            self.checkpoint_saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_loss': self.best_loss,
                'optimizer': self.optimizer.state_dict(),
            }, is_best)

            # Update the learning rate
            self.lr_scheduler.step()

    def do_tracking(self, search1, search2, template):
        # template, search1, search2 shape: batch_size, channels, height, width

        template = template.cuda(non_blocking=True)
        search1 = search1.cuda(non_blocking=True)
        search2 = search2.cuda(non_blocking=True)
        if self.gpu_num > 1:
            template_feat = self.model.module.feature(template)
            search1_feat = self.model.module.feature(search1)
            search2_feat = self.model.module.feature(search2)
        else:
            template_feat = self.model.feature(template)
            search1_feat = self.model.feature(search1)
            search2_feat = self.model.feature(search2)

        # forward tracking 1
        # compute the model response for the template, search region and label
        with torch.no_grad():
            response1 = self.model(template_feat, search1_feat, self.label)
        fake_yf = fft.rfftn(create_fake_y(self.initial_y, response1), dim=[-2, -1]).cuda(non_blocking=True)

        # forward tracking 2
        # compute the model response for the template, search region and label
        with torch.no_grad():
            response2 = self.model(search1_feat, search2_feat, fake_yf)
        fake_yf = fft.rfftn(create_fake_y(self.initial_y, response2), dim=[-2, -1]).cuda(non_blocking=True)

        # backward tracking
        output = self.model(search2_feat, template_feat, fake_yf)

        # the sample dropout is necessary, otherwise we find the loss tends to become unstable
        output = output_drop(output, self.target)

        return output

    def train(self, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        self.model.train()

        end = time.time()
        for i, (template, search1, search2) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            output = self.do_tracking(search1, search2, template)
            # consistency loss. target is the initial Gaussian label
            loss = self.criterion(output, self.target) / self.batch_size
            # measure accuracy and record loss
            losses.update(loss.item())

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.args.print_freq == 0:
                print(f'Epoch: [{epoch}][{i}/{len(self.train_loader)}]\t'
                      f'Time {batch_time.val:2.3f} ({batch_time.avg:2.3f})\t'
                      f'Data {data_time.val:2.3f} ({data_time.avg:2.3f})\t'
                      f'Loss {losses.val:2.4f} ({losses.avg:2.4f})\t')

    def validate(self):
        batch_time = AverageMeter()
        losses = AverageMeter()

        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (template, search1, search2) in enumerate(self.val_loader):

                output = self.do_tracking(search1, search2, template)
                loss = self.criterion(output, self.target) / self.batch_size

                # measure accuracy and record loss
                losses.update(loss.item())

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.args.print_freq == 0:
                    print(f'Test: [{i}/{len(self.val_loader)}]\t'
                          f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          f'Loss {losses.val:.4f} ({losses.avg:.4f})\t')

            print(' * Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses))

        return losses.avg

    pass


def main(args, gpu_num, train_loader, val_loader):
    trainer = UdtTrainer(args, gpu_num, train_loader, val_loader)
    trainer()
