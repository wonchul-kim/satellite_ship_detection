from glob import glob 
import os.path as osp 
import os
import numpy as np
import cv2
from shutil import copyfile
from tqdm import tqdm

modes = ['train', 'val']
labels = ['ship']

for mode in modes:
    input_dir = '/HDD/datasets/public/dota/v1'
    output_dir = '/HDD/datasets/public/dota/v1_ship'

    if not osp.exists(output_dir):
        os.mkdir(output_dir)
        
    if not osp.exists(osp.join(output_dir, 'images', mode)):
        os.makedirs(osp.join(output_dir, 'images', mode))
        
    if not osp.exists(osp.join(output_dir, 'labelTxt', mode)):
        os.makedirs(osp.join(output_dir, 'labelTxt', mode))
        
    label_files = glob(osp.join(input_dir, f'labelTxt/{mode}/*.txt'))
    xs, ys = [], []
    rxs, rys = [], []
    for label_file in tqdm(label_files, desc=mode):
        img = None
        filename = osp.split(osp.splitext(label_file)[0])[-1]
        lf = open(label_file, 'r')
        save = False 
        new_labels = []
        while True:
            line = lf.readline()
            
            if not line: break 
            
            if not line.startswith('acquisition') and not line.startswith('imagesource:') and not line.startswith('gsd:'):
                if 'ship' in line:
                    save = True
                    new_labels.append(line)

                    anns = line.split(" ")
                    _xs, _ys = [], []
                    for idx, ann in enumerate(anns[:8]):
                        idx += 1
                        if idx%2 == 0: 
                            _xs.append(float(ann))
                        else:
                            _ys.append(float(ann))
                    width = np.max(_xs) - np.min(_xs)
                    height = np.max(_ys) - np.min(_ys)
                    xs.append(width)
                    ys.append(height)
                    
                    if img is None:
                        img_file = osp.join(input_dir, 'images', mode, filename + '.png')
                        img = cv2.imread(img_file)
                        img_h, img_w, img_c = img.shape
                        rxs.append(width/img_w)
                        rys.append(height/img_h)
            else:
                new_labels.append(line)
                
        lf.close()
        
        if img is not None:
            copyfile(img_file, osp.join(output_dir, 'images', mode, filename + '.png'))
        
        if save:
            f = open(osp.join(output_dir, 'labelTxt', mode, osp.split(label_file)[-1]), 'w')
            for new_label in new_labels:
                f.write(new_label)
            f.close()
            
    f = open(osp.join(output_dir, f'{mode}_info.txt'), 'w')
    f.write(f'mean x: {np.mean(xs)}\n')
    f.write(f'mean y: {np.mean(ys)}\n')
    f.write(f'max x: {np.max(xs)}\n')
    f.write(f'max y: {np.max(ys)}\n')
    f.write(f'min x: {np.min(xs)}\n')
    f.write(f'min y: {np.min(ys)}\n')
    f.write(f'mean ratio x by image: {np.mean(rxs)}\n')
    f.write(f'mean ratio y by image: {np.mean(rys)}\n')
    f.write(f'max ratio x by image: {np.max(rxs)}\n')
    f.write(f'max ratio y by image: {np.max(rys)}\n')
    f.write(f'min ratio x by image: {np.min(rxs)}\n')
    f.write(f'min ratio y by image: {np.min(rys)}\n')
    f.close()

    
        
    

              
        




