import os
import os.path as osp 
import glob 
import json
import numpy as np
from shutil import copyfile
from tqdm import tqdm

input_dir = '/HDD/datasets/public/SODA/SODA_A'
output_dir = '/HDD/datasets/public/SODA/SODA_A/ship_dota'

if not osp.exists(output_dir):
    os.mkdir(output_dir)
    
if not osp.exists(osp.join(output_dir, 'images')):
    os.mkdir(osp.join(output_dir, 'images'))
    
if not osp.exists(osp.join(output_dir, 'labels')):
    os.mkdir(osp.join(output_dir, 'labels'))
    
ann_files = glob.glob(osp.join(input_dir, 'Annotations/all/*.json'))
print(f"There are {len(ann_files)} files")

cnt = 0
for ann_file in tqdm(ann_files):
    with open(ann_file, 'r') as jf:
        anns = json.load(jf)
    filename = anns['images']['file_name']
    categories = anns['categories']
    annotations = anns['annotations']

    is_ship_included = False
    dota_anns = []
    for annotation in annotations:
        points = np.array(annotation['poly'], np.int32)
        index = annotation['category_id']
        label = categories[index]['name']
        
        if label == 'ship':
            is_ship_included = True
            str_points = ""
            for point in points:
                str_points += f"{str(point)} "
            str_points += f"{label} 0"
            dota_anns.append(str_points)
                
            
        
    if is_ship_included:
        img_file = osp.join(input_dir, f'Images/{filename}')
        assert osp.exists(img_file), ValueError(f"There is no such image: {img_file}")

        copyfile(img_file, osp.join(output_dir, f'images/{filename}'))
        txt = open(osp.join(output_dir, f'labels/{filename.split(".")[0]}.txt'), 'w')
        for dota_ann in dota_anns:
            txt.write(dota_ann)
            txt.write("\n")
        cnt += 1
        
        
print(f"There are {cnt} ship")
                
        
        
        
        