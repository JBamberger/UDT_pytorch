import os

import torch.utils.data as data
from os.path import join
import cv2
import json
import numpy as np

import config


class ILSVRC2015(data.Dataset):
    def __init__(self, file=None, root=None, range=10, train=True):
        """
        range: number of frames to select search templates from
        """

        if file is None:
            file = os.path.join(config.dataset_root, 'ILSVRC2015', 'dataset.json')

        if root is None:
            root = os.path.join(config.dataset_root, 'ILSVRC2015', 'crop_125_2.0')

        self.imdb = json.load(open(file, 'r'))
        self.root = root
        self.range = range
        self.train = train
        self.mean = np.array([109, 120, 119], dtype=np.float32).reshape([3, 1, 1])

    def _load_frame(self, index: int):
        img = cv2.imread(join(self.root, f'{index:08d}.jpg'))
        # TODO: Frames are not converted to RGB and range 0-1 or -0.5, 0.5. Is this an error or is there a reason?
        # Bring color axis to front. Shape is c, h, w
        return np.transpose(img, (2, 0, 1)).astype(np.float32) - self.mean

    def __getitem__(self, item):
        """
        output shapes:
        3 x h x w
        """
        if self.train:
            target_id = self.imdb['train_set'][item]
        else:
            target_id = self.imdb['val_set'][item]

        # range_down = self.imdb['down_index'][target_id]
        # search_id = np.random.randint(-min(range_down, self.range), min(range_up, self.range)) + target_id

        range_up = self.imdb['up_index'][target_id]
        search_id1 = np.random.randint(1, min(range_up, self.range + 1)) + target_id
        search_id2 = np.random.randint(1, min(range_up, self.range + 1)) + target_id

        target = self._load_frame(target_id)
        search1 = self._load_frame(search_id1)
        search2 = self._load_frame(search_id2)

        return target, search1, search2

    def __len__(self):
        if self.train:
            return len(self.imdb['train_set'])
        else:
            return len(self.imdb['val_set'])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    data = ILSVRC2015(train=True)
    n = len(data)
    fig = plt.figure(1)
    ax = fig.add_axes([0, 0, 1, 1])

    for i in range(n):
        z, x, _ = data[i]
        z, x = np.transpose(z, (1, 2, 0)).astype(np.uint8), np.transpose(x, (1, 2, 0)).astype(np.uint8)
        zx = np.concatenate((z, x), axis=1)

        ax.imshow(cv2.cvtColor(zx, cv2.COLOR_BGR2RGB))
        p = patches.Rectangle(
            (125 / 3, 125 / 3), 125 / 3, 125 / 3, fill=False, clip_on=False, linewidth=2, edgecolor='g')
        ax.add_patch(p)
        p = patches.Rectangle(
            (125 / 3 + 125, 125 / 3), 125 / 3, 125 / 3, fill=False, clip_on=False, linewidth=2, edgecolor='r')
        ax.add_patch(p)
        plt.pause(0.5)
