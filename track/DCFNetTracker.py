import argparse
import time as time
from os import makedirs
from os.path import join, isdir

import cv2
import numpy as np
import torch

import config as cfg
from dataset import OtbDataset
from track.DCFNet import DCFNet
from track.TrackerConfig import TrackerConfig
from track.eval_otb import eval_auc
from util import crop_chw, cxy_wh_2_rect1, rect1_2_cxy_wh, cxy_wh_2_bbox


class DCFNetTracker(object):
    def __init__(self, im, init_rect, config=TrackerConfig(), gpu=True):
        self.gpu = gpu
        self.config = config
        self.net = DCFNet(config)
        self.net.load_param(config.feature_path)
        self.net.eval()
        if gpu:
            self.net.cuda()

        # confine results
        target_pos, target_sz = rect1_2_cxy_wh(init_rect)
        self.min_sz = np.maximum(config.min_scale_factor * target_sz, 4)
        self.max_sz = np.minimum(im.shape[:2], config.max_scale_factor * target_sz)

        # crop template
        window_sz = target_sz * (1 + config.padding)
        bbox = cxy_wh_2_bbox(target_pos, window_sz)
        patch = crop_chw(im, bbox, self.config.crop_sz)

        target = patch - config.net_average_image
        self.net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda())
        self.target_pos, self.target_sz = target_pos, target_sz
        self.patch_crop = np.zeros((config.num_scale, patch.shape[0], patch.shape[1], patch.shape[2]),
                                   np.float32)  # buff

    def track(self, im):
        for i in range(self.config.num_scale):  # crop multi-scale search region
            window_sz = self.target_sz * (self.config.scale_factor[i] * (1 + self.config.padding))
            bbox = cxy_wh_2_bbox(self.target_pos, window_sz)
            self.patch_crop[i, :] = crop_chw(im, bbox, self.config.crop_sz)

        search = self.patch_crop - self.config.net_average_image

        if self.gpu:
            response = self.net(torch.Tensor(search).cuda())
        else:
            response = self.net(torch.Tensor(search))
        peak, idx = torch.max(response.view(self.config.num_scale, -1), 1)
        peak = peak.data.cpu().numpy() * self.config.scale_penalties
        best_scale = np.argmax(peak)
        r_max, c_max = np.unravel_index(idx[best_scale], self.config.net_input_size)

        if r_max > self.config.net_input_size[0] / 2:
            r_max = r_max - self.config.net_input_size[0]
        if c_max > self.config.net_input_size[1] / 2:
            c_max = c_max - self.config.net_input_size[1]
        window_sz = self.target_sz * (self.config.scale_factor[best_scale] * (1 + self.config.padding))

        self.target_pos = self.target_pos + np.array([c_max, r_max]) * window_sz / self.config.net_input_size
        self.target_sz = np.minimum(np.maximum(window_sz / (1 + self.config.padding), self.min_sz), self.max_sz)

        # model update
        window_sz = self.target_sz * (1 + self.config.padding)
        bbox = cxy_wh_2_bbox(self.target_pos, window_sz)
        patch = crop_chw(im, bbox, self.config.crop_sz)
        target = patch - self.config.net_average_image
        self.net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda(), lr=self.config.interp_factor)

        return cxy_wh_2_rect1(self.target_pos, self.target_sz)  # 1-index


if __name__ == '__main__':
    # base dataset path and setting
    parser = argparse.ArgumentParser(description='Test DCFNet on OTB')
    parser.add_argument('--dataset', metavar='SET', default='OTB2013',
                        choices=['OTB2013', 'OTB2015'], help='tune on which dataset')
    parser.add_argument('--model', metavar='PATH', default='param.pth')
    args = parser.parse_args()

    ds = OtbDataset(variant=args.dataset)

    use_gpu = True
    visualization = False

    # default parameter and load feature extractor network
    config = TrackerConfig()
    net = DCFNet(config)
    net.load_param(args.model)
    net.eval().cuda()

    speed = []
    for video_id, video in enumerate(ds):  # run without resetting
        tic = time.time()

        n_images = len(video)
        target_pos, target_sz = rect1_2_cxy_wh(video.init_rect)  # OTB label is 1-indexed

        im = video.frame_at(0)  # HxWxC

        # confine results
        min_sz = np.maximum(config.min_scale_factor * target_sz, 4)
        max_sz = np.minimum(im.shape[:2], config.max_scale_factor * target_sz)

        # crop template
        window_sz = target_sz * (1 + config.padding)
        bbox = cxy_wh_2_bbox(target_pos, window_sz)
        patch = crop_chw(im, bbox, config.crop_sz)

        target = patch - config.net_average_image
        net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda())

        res = [cxy_wh_2_rect1(target_pos, target_sz)]  # save in .txt
        patch_crop = np.zeros((config.num_scale, patch.shape[0], patch.shape[1], patch.shape[2]), np.float32)
        for f in range(1, n_images):  # track
            im = video.frame_at(f)

            for i in range(config.num_scale):  # crop multi-scale search region
                window_sz = target_sz * (config.scale_factor[i] * (1 + config.padding))
                bbox = cxy_wh_2_bbox(target_pos, window_sz)
                patch_crop[i, :] = crop_chw(im, bbox, config.crop_sz)

            search = patch_crop - config.net_average_image
            search = torch.Tensor(search).cuda()

            response = net(search)

            peak, idx = torch.max(response.view(config.num_scale, -1), 1)
            idx = idx.cpu().numpy()
            peak = peak.data.cpu().numpy() * config.scale_penalties
            best_scale = np.argmax(peak)
            r_max, c_max = np.unravel_index(idx[best_scale], config.net_input_size)

            if r_max > config.net_input_size[0] / 2:
                r_max = r_max - config.net_input_size[0]
            if c_max > config.net_input_size[1] / 2:
                c_max = c_max - config.net_input_size[1]
            window_sz = target_sz * (config.scale_factor[best_scale] * (1 + config.padding))

            target_pos = target_pos + np.array([c_max, r_max]) * window_sz / config.net_input_size
            target_sz = np.minimum(np.maximum(window_sz / (1 + config.padding), min_sz), max_sz)

            # model update
            window_sz = target_sz * (1 + config.padding)
            bbox = cxy_wh_2_bbox(target_pos, window_sz)
            patch = crop_chw(im, bbox, config.crop_sz)
            target = patch - config.net_average_image
            net.update(torch.Tensor(np.expand_dims(target, axis=0)).cuda(), lr=config.interp_factor)

            res.append(cxy_wh_2_rect1(target_pos, target_sz))  # 1-index

            if visualization:
                im_show = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                cv2.rectangle(im_show, (int(target_pos[0] - target_sz[0] / 2), int(target_pos[1] - target_sz[1] / 2)),
                              (int(target_pos[0] + target_sz[0] / 2), int(target_pos[1] + target_sz[1] / 2)),
                              (0, 255, 0), 3)
                cv2.putText(im_show, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(video.video_name, im_show)
                cv2.waitKey(1)

        toc = time.time() - tic
        fps = n_images / toc
        speed.append(fps)
        print(f'{video_id:3d} Video: {video.video_name:12s} Time: {toc:3.1f}s\tSpeed: {fps:3.1f}fps')

        # save result
        test_path = join(cfg.results_root, args.dataset, 'DCFNet_test')
        if not isdir(test_path):
            makedirs(test_path)

        result_path = join(test_path, video.video_name + '.txt')
        with open(result_path, 'w') as f:
            for x in res:
                f.write(','.join(['{:.2f}'.format(i) for i in x]) + '\n')

    print('***Total Mean Speed: {:3.1f} (FPS)***'.format(np.mean(speed)))

    eval_auc(args.dataset, 'DCFNet_test', 0, 1)
