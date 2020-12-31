import os
import shutil
from os.path import join

import cv2
import numpy as np
import torch


def cxy_wh_2_rect1(pos, sz):
    return np.array([pos[0] - sz[0] / 2 + 1, pos[1] - sz[1] / 2 + 1, sz[0], sz[1]])  # 1-index


def rect1_2_cxy_wh(rect):
    return np.array([rect[0] + rect[2] / 2 - 1, rect[1] + rect[3] / 2 - 1]), np.array([rect[2], rect[3]])  # 0-index


def cxy_wh_2_bbox(cxy, wh):
    return np.array([cxy[0] - wh[0] / 2, cxy[1] - wh[1] / 2, cxy[0] + wh[0] / 2, cxy[1] + wh[1] / 2])  # 0-index


def gaussian_shaped_labels(sigma, sz):
    x, y = np.meshgrid(np.arange(1, sz[0] + 1) - np.floor(float(sz[0]) / 2),
                       np.arange(1, sz[1] + 1) - np.floor(float(sz[1]) / 2))
    d = x ** 2 + y ** 2
    g = np.exp(-0.5 / (sigma ** 2) * d)
    g = np.roll(g, int(-np.floor(float(sz[0]) / 2.) + 1), axis=0)
    g = np.roll(g, int(-np.floor(float(sz[1]) / 2.) + 1), axis=1)
    return g


def crop_chw(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return np.transpose(crop, (2, 0, 1))


if __name__ == '__main__':
    a = gaussian_shaped_labels(10, [5, 5])
    print(a)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def output_drop(output, target):
    delta1 = (output - target) ** 2
    batch_sz = delta1.shape[0]
    delta = delta1.view(batch_sz, -1).sum(dim=1)
    sort_delta, index = torch.sort(delta, descending=True)
    # unreliable samples (10% of the total) do not produce grad (we simply copy the groundtruth label)
    for i in range(int(round(0.1 * batch_sz))):
        output[index[i], ...] = target[index[i], ...]
    return output


def compute_lr_gamma(initial_lr, final_lr, epochs):
    return (final_lr / initial_lr) ** (1 / epochs)


class CheckpointSaver:
    def __init__(self, save_path, verbose=False):
        self.save_path = save_path
        self.verbose = verbose

    def save_checkpoint(self, state, is_best):
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
            if self.verbose:
                print("Creating checkpoint directory: " + self.save_path)

        cp_file = os.path.join(self.save_path, 'checkpoint.pth.tar')
        torch.save(state, cp_file)
        if self.verbose:
            print("Writing checkpoint file: " + cp_file)

        if is_best:
            best_cp_file = os.path.join(self.save_path, 'model_best.pth.tar')
            shutil.copyfile(cp_file, best_cp_file)

            if self.verbose:
                print("Creating copy for best checkpoint: " + best_cp_file)


def create_fake_y(initial_y, response):
    batch_size = response.shape[0]
    d = response.shape[2]

    # compute the coordinates of the maximum for each batch image
    m = response.view(batch_size, -1).argmax(dim=1).view(-1, 1)
    cy = m // d
    cx = m % d

    # roll the template to the given maximum for each batch image
    fake_y = torch.empty_like(response)
    for i in range(batch_size):
        fake_y[i, ...] = torch.roll(initial_y, shifts=(cy[i].item(), cx[i].item()), dims=(0, 1))

    return fake_y
