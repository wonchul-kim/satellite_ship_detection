from ultralytics import YOLO 
from pathlib import Path
ROOT = Path(__file__).resolve()
FILE = ROOT.parent

import os.path as osp
import glob 
import cv2

model = YOLO(FILE / "yolox-obb.pt")


input_dir = '/HDD/datasets/public/dota/ship/v1.5_ship_yolo_obb_split_640_100/images/val'
input_img_ext = 'jpg'
imgsz = 640
conf_threshold = 0.3
img_files = glob.glob(osp.join(input_dir, f'*.{input_img_ext}'))

for img_file in img_files:
    preds = model(img_file, save=True, imgsz=imgsz, conf=conf_threshold, verbose=True, device='1')[0]
    
    idx2class = preds.names
    obb_result = preds.obb
    classes = obb_result.cls.tolist()
    confs = obb_result.conf.tolist()
    
    if len(obb_result) != 0:
        print(obb_result)
        