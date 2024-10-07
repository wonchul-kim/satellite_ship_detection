from ultralytics import YOLO 
from pathlib import Path
ROOT = Path(__file__).resolve()
FILE = ROOT.parent

model = YOLO("yolo11x-obb.pt")
results = model.train(data=FILE / "dotav1.5.yaml", 
                      epochs=200, imgsz=1024, device='0,1,2,3', batch=16,
                      lrf=0.001, degrees=45, flipud=0.25, fliplr=0.25, val=False,
                      )