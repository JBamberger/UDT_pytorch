from os.path import join, isdir
import os
import argparse
import numpy as np
import json
import cv2

from tqdm import tqdm


def crop_patch(im, img_sz):
    w = float((img_sz[0] / 2) * (1 + args.padding))
    h = float((img_sz[1] / 2) * (1 + args.padding))
    x = float((img_sz[0] / 2) - w / 2)
    y = float((img_sz[1] / 2) - h / 2)

    a = (args.output_size - 1) / w
    b = (args.output_size - 1) / h
    c = -a * x
    d = -b * y

    mapping = np.array([[a, 0, c],
                        [0, b, d]], dtype=np.float)

    return cv2.warpAffine(im, mapping, (args.output_size, args.output_size),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0))


def main():
    os.chdir(args.base_path)

    vid = json.load(open('vid.json', 'r'))
    num_all_frame = 1298523
    num_val = 3000
    # crop image
    lmdb = {
        'down_index': np.zeros(num_all_frame, np.int),  # buff
        'up_index': np.zeros(num_all_frame, np.int),
    }
    crop_base_path = f'crop_{args.output_size:d}_{args.padding:1.1f}'
    if not isdir(crop_base_path):
        os.mkdir(crop_base_path)
    count = 0

    with open("log.txt", "w", encoding="utf8") as logf:
        for subset in vid:
            total = 0
            for v in subset:
                total += len(v['frame'])

            progress = tqdm(total=total)
            for video in subset:
                frames = video['frame']
                n_frames = len(frames)
                for f, frame in enumerate(frames):
                    img_path = join(video['base_path'], frame['img_path'])
                    out_path = join(crop_base_path, '{:08d}.jpg'.format(count))

                    if not os.path.exists(out_path):
                        # read, crop, write
                        cv2.imwrite(out_path, crop_patch(cv2.imread(img_path), frame['frame_sz']))
                        logf.write("processed ")
                        logf.write(out_path)
                        logf.write('\n')
                    else:
                        logf.write("skipped ")
                        logf.write(out_path)
                        logf.write('\n')

                    # how many frames to the first frame
                    lmdb['down_index'][count] = f
                    # how many frames to the last frame
                    lmdb['up_index'][count] = n_frames - f
                    count += 1
                    progress.update()

        template_id = np.where(lmdb['up_index'] > 1)[0]  # NEVER use the last frame as template! I do not like bidirectional
        rand_split = np.random.choice(len(template_id), len(template_id))
        lmdb['train_set'] = template_id[rand_split[:(len(template_id) - num_val)]]
        lmdb['val_set'] = template_id[rand_split[(len(template_id) - num_val):]]
        print(len(lmdb['train_set']))
        print(len(lmdb['val_set']))
        # to list for json
        lmdb['train_set'] = lmdb['train_set'].tolist()
        lmdb['val_set'] = lmdb['val_set'].tolist()
        lmdb['down_index'] = lmdb['down_index'].tolist()
        lmdb['up_index'] = lmdb['up_index'].tolist()

        print('lmdb json, please wait 5 seconds~')
        json.dump(lmdb, open('dataset.json', 'w'), indent=2)
        print('done!')


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Generate training data (cropped) for DCFNet_pytorch')
    parse.add_argument('-d', '--dir', dest='base_path',required=True, type=str, help='working directory')
    parse.add_argument('-v', '--visual', dest='visual', action='store_true', help='whether visualise crop')
    parse.add_argument('-o', '--output_size', dest='output_size', default=125, type=int, help='crop output size')
    parse.add_argument('-p', '--padding', dest='padding', default=2, type=float, help='crop padding size')

    args = parse.parse_args()

    print(args)

    main()
