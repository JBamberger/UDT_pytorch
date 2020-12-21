import argparse
import glob
import json
import os
import xml.etree.ElementTree as ET
from os import listdir
from os.path import join

from tqdm import tqdm

OUT_FILE = 'vid.json'

parser = argparse.ArgumentParser(description='Parse the VID Annotations for training DCFNet')
parser.add_argument('data', metavar='DIR', help='path to VID')
args = parser.parse_args()

print('VID2015 Data:')
VID_base_path = args.data
ann_base_path = join(VID_base_path, 'Annotations', 'VID', 'train')
img_base_path = join(VID_base_path, 'Data', 'VID', 'train')
# sub_sets = sorted({'a', 'b', 'c', 'd', 'e'})
sub_sets = sorted({'ILSVRC2015_VID_train_0000',
                   'ILSVRC2015_VID_train_0001',
                   'ILSVRC2015_VID_train_0002',
                   'ILSVRC2015_VID_train_0003'})
total_frame = 0

vid = []
for subset_name in sub_sets:
    subset_path = join(ann_base_path, subset_name)
    video_paths = sorted(listdir(subset_path))

    print(f'Subset: {subset_name}')
    vids = tqdm(enumerate(video_paths), desc='Processing videos:', total=len(video_paths))

    s = []
    for vi, video in vids:
        v = {
            'base_path': join(img_base_path, subset_name, video),
            'frame': []
        }
        frame_annotation_paths = sorted(glob.glob(join(subset_path, video, '*.xml')))
        for frame_annotation_path in frame_annotation_paths:
            total_frame += 1
            xmltree = ET.parse(frame_annotation_path)
            f = {
                'frame_sz': [int(it.text) for it in (xmltree.findall('size')[0])],
                'img_path': os.path.splitext(os.path.basename(frame_annotation_path))[0] + '.JPEG',
            }
            v['frame'].append(f)
        s.append(v)
    vid.append(s)

print(f'Total frame number: {total_frame:d}')
print('Writing video information, please wait...')

with open(OUT_FILE, 'w') as of:
    json.dump(vid, of, indent=2)

print(f'Wrote output to {os.path.abspath(OUT_FILE)}.')
