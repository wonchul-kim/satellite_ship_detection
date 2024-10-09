from ultralytics import YOLO 
from pathlib import Path
ROOT = Path(__file__).resolve()
FILE = ROOT.parent

model = YOLO("yolo11x-obb.pt")
results = model.train(data=FILE / "dotav1.yaml", 
                      epochs=500, imgsz=640, device='0,1,2,3', batch=64,
                      lrf=0.001, degrees=45, flipud=0.25, fliplr=0.25, val=False,
                      )