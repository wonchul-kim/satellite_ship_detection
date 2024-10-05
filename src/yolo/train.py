from ultralytics import YOLO 
from pathlib import Path
ROOT = Path(__file__).resolve()
FILE = ROOT.parent

model = YOLO("yolo11l-obb.pt")
results = model.train(data=FILE / "dotav1.yaml", 
                      epochs=300, imgsz=1024, device='0,1', batch=2,
                      multi_scale=True, 
                      lrf=0.001, degrees=45, flipud=0.25, fliplr=0.25,
                      )