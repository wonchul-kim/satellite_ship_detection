

from glob import glob 
import os.path as osp 
import os
import numpy as np
import cv2
from shutil import copyfile
from tqdm import tqdm

labels = ['ship']

input_dir = '/HDD/datasets/public/DOTA/DOTA v2.0.v1i.yolov8-obb'
output_dir = '/HDD/datasets/public/DOTA/v2_ship_yolo_obb'

if not osp.exists(output_dir):
    os.mkdir(output_dir)
    
labels_output_dir = osp.join(output_dir, 'labels/train')
images_output_dir = osp.join(output_dir, 'images/train')
if not osp.exists(labels_output_dir):
    os.makedirs(labels_output_dir)
    
if not osp.exists(images_output_dir):
    os.makedirs(images_output_dir)
    
txt_files = glob(osp.join(input_dir, f'train/labels/*.txt'))
for txt_file in tqdm(txt_files):
    img_file = None
    filename = osp.split(osp.splitext(txt_file)[0])[-1]
    lf = open(txt_file, 'r')
    save = False 
    new_labels = []
    while True:
        line = lf.readline()
        
        if not line: break 
        
        _line = line.split(" ")
        
        if int(_line[0]) == 12:
            save = True
            _line[0] = '0'
            new_labels.append(' '.join(_line))

            if img_file is None:
                img_file = osp.join(input_dir, 'train', 'images', filename + '.jpg')
            
    lf.close()
    
    if img_file is not None:
        copyfile(img_file, osp.join(output_dir, 'images/train', filename + '.jpg'))
        f = open(osp.join(output_dir, 'labels/train', osp.split(txt_file)[-1]), 'w')
        for new_label in new_labels:
            f.write(new_label)
        f.close()


    


            
    




