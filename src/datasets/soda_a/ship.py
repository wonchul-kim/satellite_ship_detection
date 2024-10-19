import os
import os.path as osp 
import glob 
import json
import numpy as np

input_dir = '/HDD/datasets/public/SODA/SODA_A'
output_dir = '/HDD/datasets/public/SODA/SODA_A/vis'

if not osp.exists(output_dir):
    os.mkdir(output_dir)
    
ann_files = glob.glob(osp.join(input_dir, 'Annotations/all/*.json'))
print(f"There are {len(ann_files)} files")

for ann_file in ann_files:
    with open(ann_file, 'r') as jf:
        anns = json.load(jf)
    filename = anns['images']['file_name']
    categories = anns['categories']
    annotations = anns['annotations']

    import cv2
    img_file = osp.join(input_dir, f'Images/{filename}')
    assert osp.exists(img_file), ValueError(f"There is no such image: {img_file}")
    img = cv2.imread(img_file)
        
    for annotation in annotations:
        points = np.array(annotation['poly'], np.int32)
        index = annotation['id']
        label = categories[int(index)]['name']
        
        points = points.reshape((-1, 1, 2))
        
        cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=3)
        
        cv2.imwrite(osp.join(output_dir, filename), img)
        
        
                
        
        
        
        