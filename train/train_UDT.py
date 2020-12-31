import argparse

from dataset import ILSVRC2015

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
train_dataset = ILSVRC2015(train=True, range=args.range)
val_dataset = ILSVRC2015(train=False, range=args.range)

if __name__ == '__main__':
    # imports here ensure that the multiprocessing does not load these imports in every process.
    import torch
    from torch.utils.data import dataloader
    from train.train_UDT_impl import main

    print(args)
    gpu_num = torch.cuda.device_count()

    # Create training dataset loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size * gpu_num, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    # Create validation dataset loader

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size * gpu_num, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    main(args, gpu_num, train_loader, val_loader)

    # import torch.autograd.profiler as profiler
    # with profiler.profile(use_cuda=True, record_shapes=True, profile_memory=True, with_stack=True) as p:
    #     with profiler.record_function('model_inference'):
    #         main(args, gpu_num, train_loader, val_loader)
    #
    # perf = p.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=-1)
    # with open('perf.txt', 'w', encoding='utf8') as f:
    #     f.write(perf)
    # exit()
    # print(p.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=-1))
