import os.path as osp
import glob
import os
from shutil import copyfile
from tqdm import tqdm

input_dir = '/HDD/datasets/public/dota/ship/v1_ship_yolo_obb_split_640_100'
output_dir = '/HDD/datasets/public/dota/ship/v1_ship_yolo_obb_split_640_100_no_background'
if not osp.exists(output_dir):
    os.mkdir(output_dir)

modes = ['train', 'val']

for mode in modes:
    txt_files = glob.glob(osp.join(input_dir, 'labels', mode, '*.txt'))
    
    labels_output_dir = osp.join(output_dir, 'labels', mode)
    images_output_dir = osp.join(output_dir, 'images', mode)
    if not osp.exists(labels_output_dir):
        os.makedirs(labels_output_dir)
        
    if not osp.exists(images_output_dir):
        os.makedirs(images_output_dir)
    
    for txt_file in tqdm(txt_files, desc=mode):
        filename = osp.split(osp.splitext(txt_file)[0])[-1]        
        img_file = osp.join(input_dir, 'images', mode, filename + '.jpg')
        
        assert osp.exists(img_file), RuntimeError(f"There is no such image: {img_file}")
        
        copyfile(txt_file, osp.join(labels_output_dir, filename + '.txt'))
        copyfile(img_file, osp.join(images_output_dir, filename + '.jpg'))
        
