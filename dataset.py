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

        # training data
        # crop_base_path = os.path.join(config.dataset_root,
        #     'ILSVRC2015', f'crop_{args.input_sz:d}_{args.padding:1.1f}')
        # if not isdir(crop_base_path):
        #     print(f'please run gen_training_data.py --output_size {args.input_sz:d} --padding {args.padding:.1f}!')
        #     exit()

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


OTB2013 = {'carDark', 'car4', 'david', 'david2', 'sylvester', 'trellis', 'fish', 'mhyang', 'soccer', 'matrix',
           'ironman', 'deer', 'skating1', 'shaking', 'singer1', 'singer2', 'coke', 'bolt', 'boy', 'dudek',
           'crossing', 'couple', 'football1', 'jogging_1', 'jogging_2', 'doll', 'girl', 'walking2', 'walking',
           'fleetface', 'freeman1', 'freeman3', 'freeman4', 'david3', 'jumping', 'carScale', 'skiing', 'dog1',
           'suv', 'motorRolling', 'mountainBike', 'lemming', 'liquor', 'woman', 'faceocc1', 'faceocc2',
           'basketball', 'football', 'subway', 'tiger1', 'tiger2'}

OTB2015 = {'carDark', 'car4', 'david', 'david2', 'sylvester', 'trellis', 'fish', 'mhyang', 'soccer', 'matrix',
           'ironman', 'deer', 'skating1', 'shaking', 'singer1', 'singer2', 'coke', 'bolt', 'boy', 'dudek',
           'crossing', 'couple', 'football1', 'jogging_1', 'jogging_2', 'doll', 'girl', 'walking2', 'walking',
           'fleetface', 'freeman1', 'freeman3', 'freeman4', 'david3', 'jumping', 'carScale', 'skiing', 'dog1',
           'suv', 'motorRolling', 'mountainBike', 'lemming', 'liquor', 'woman', 'faceocc1', 'faceocc2',
           'basketball', 'football', 'subway', 'tiger1', 'tiger2', 'clifBar', 'biker', 'bird1', 'blurBody',
           'blurCar2', 'blurFace', 'blurOwl', 'box', 'car1', 'crowds', 'diving', 'dragonBaby', 'human3', 'human4_2',
           'human6', 'human9', 'jump', 'panda', 'redTeam', 'skating2_1', 'skating2_2', 'surfer', 'bird2',
           'blurCar1', 'blurCar3', 'blurCar4', 'board', 'bolt2', 'car2', 'car24', 'coupon', 'dancer', 'dancer2',
           'dog', 'girl2', 'gym', 'human2', 'human5', 'human7', 'human8', 'kiteSurf', 'man', 'rubik', 'skater',
           'skater2', 'toy', 'trans', 'twinnings', 'vase'}

OTB_sequences = {
    'OTB2013': OTB2013,
    'OTB2015': OTB2015
}


class OtbVideo:

    def __init__(self, key, video_name, image_files, init_rect, gt_rects):
        self.key = key
        self.video_name = video_name
        self.init_rect = np.array(init_rect).astype(np.float)
        self.image_files = image_files
        self._gt_rects = gt_rects
        self.length = len(self.image_files)

    @property
    def gt_rects(self):
        return np.array(self._gt_rects).astype(np.float)

    def contained_in(self, dataset_name):
        return self.key in OTB_sequences[dataset_name]

    def frame_at(self, index):
        assert 0 <= index < self.length

        path = self.image_files[index]
        frame = cv2.imread(path)

        if frame.data is None:
            raise AssertionError('Video file not fount at path {path}')

        return frame

    def __len__(self):
        return self.length


class OtbDataset:
    def __init__(self, variant='OTB2015', dataset_path=None):

        if dataset_path is None:
            dataset_path = os.path.join(config.dataset_root, 'OTB')
        self.dataset_path = dataset_path

        with open(os.path.join(dataset_path, f'{variant}.json'), 'r') as annotation_file:
            self.annotations = json.load(annotation_file)
        self.videos = sorted(self.annotations.keys())

    def __iter__(self):
        """
        Schema:
        { 'key': {
            'name': folderName
            'image_files': [imageName1, ...]
            'init_rect: [int,int,int,int]
            'gt_rect': [
                    [int,int,int,int],
                    ...
                ]
            }
        }
        """
        for video_key in self.videos:
            video = self.annotations[video_key]

            video_name = video['name']
            init_rect = video['init_rect']
            image_names = video['image_files']
            image_files = [os.path.join(self.dataset_path, video_name, 'img', im_f) for im_f in image_names]
            gt_rects = video['gt_rect']

            yield OtbVideo(video_key, video_name, image_files, init_rect, gt_rects)

    def __len__(self):
        return len(self.videos)


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
