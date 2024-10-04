from glob import glob 
import os.path as osp
import os
from ultralytics import YOLO
import math
import csv
import cv2
from pathlib import Path
from tqdm import tqdm
ROOT = Path(__file__).resolve()
FILE = ROOT.parent

def make_submission(input_dir='/HDD/datasets/dota/val/images/part1-001/images',
                    patch_imgsz=512,
                    confidence_threshold=6):

    patch_overlap_ratio = 0.1
    dx = int((1. - patch_overlap_ratio) * patch_imgsz)
    dy = int((1. - patch_overlap_ratio) * patch_imgsz)

    model = YOLO(FILE / Path("yolox-obb.pt"))
    # model = YOLO("yolo11n-obb.pt")
    
    img_files = glob(osp.join(input_dir, '*.png'))
    print(os.listdir(input_dir))
    print(img_files)

    results = []
    for img_file in img_files[0:1]:
        img = cv2.imread(img_file)
        img_h, img_w, img_c = img.shape
        cnt = 0
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
                output = model(patch, save=False, imgsz=patch_imgsz, conf=confidence_threshold, verbose=False)[0]
                obb_result = output.obb
                classes = obb_result.cls.tolist()
                xywhrs = obb_result.xywhr
                if 1 in classes:
                    result = {'image_name': osp.split(img_file)[-1]}
                    for (cls, xywhr) in zip(classes, xywhrs):
                        if cls == 1:
                            cx, cy, width, height, angle = xywhr
                            result.update({'cx': cx.item() + xmin, 'cy': cy.item() + ymin, 
                                        'width': width.item(), 'height': height.item(), 
                                        'angle': math.degrees(angle.item())})
                            # cv2.imwrite(f"/HDD/etc/ship_result/patch_{cnt}.png", patch)
                            # cv2.imwrite(f"/HDD/etc/ship_result/output_{cnt}.png", output.plot())
                            # cnt += 1
                    results.append(result)
        
    with open(FILE / Path("submission.csv"), mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['image_name', 'cx', 'cy', 'width', 'height', 'angle'])
        writer.writeheader()
        writer.writerows(results)
        
if __name__ == '__main__':
    make_submission('/HDD/etc/ship')