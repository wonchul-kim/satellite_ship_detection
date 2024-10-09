import numpy as np
from glob import glob
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import os.path as osp
import os
from ultralytics import YOLO
import math
import csv
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve()
FILE = ROOT.parent

def make_submission(input_dir='/HDD/datasets/dota/val/images/part1-001/images',
                    patch_imgsz=200, imgsz=640, 
                    confidence_threshold=.3):

    patch_overlap_ratio = 0.1
    dx = int((1. - patch_overlap_ratio) * patch_imgsz)
    dy = int((1. - patch_overlap_ratio) * patch_imgsz)

    model = YOLO(FILE / Path("yolox-obb.pt"))
    img_files = glob(f'{input_dir}/**/*.png', recursive=True)
    print("input_dir: ", input_dir)
    print(img_files)
    
    results = []
    for idx, img_file in enumerate(img_files):
        image_name = osp.split(img_file)[-1]
        print(f"{idx}: {image_name}")
        with Image.open(img_file) as image:
            img = np.array(image, dtype=np.uint8)

        img_h, img_w, img_c = img.shape
        print(f"   - h, w: {img_h}, {img_w}")
        for y0 in tqdm(range(0, img_h, dy)):
            for x0 in range(0, img_w, dx):
                if y0 + patch_imgsz > img_h:
                    # skip if too much overlap (> 0.6)
                    y = img_h - patch_imgsz
                else:
                    y = y0

                if x0 + patch_imgsz > img_w:
                    x = img_w - patch_imgsz
                else:
                    x = x0

                xmin, xmax, ymin, ymax = x, x + patch_imgsz, y, y + patch_imgsz
                patch = img[ymin:ymax, xmin:xmax, :]
                output = model(patch, save=True, imgsz=imgsz, conf=confidence_threshold, verbose=False,
                               show_labels=False, show_conf=False)[0]  
                obb_result = output.obb
                classes = obb_result.cls.tolist()
                xywhrs = obb_result.xywhr
                if 0 in classes:
                    for (cls, xywhr) in zip(classes, xywhrs):
                        if cls == 0:
                            result = {'image_name': osp.split(img_file)[-1]}
                            cx, cy, width, height, angle = xywhr
                            result.update({'cx': cx.item() + xmin, 'cy': cy.item() + ymin, 
                                        'width': width.item(), 'height': height.item(), 
                                        'angle': np.rad2deg(angle.item())})
                        results.append(result)
        
    with open(FILE / Path("submission.csv"), mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['image_name', 'cx', 'cy', 'width', 'height', 'angle'])
        writer.writeheader()
        writer.writerows(results)
        
if __name__ == '__main__':
    make_submission('/HDD/etc/ship')