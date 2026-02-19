import torch
#from executorch.exir import to_edge
from PIL import Image
import numpy as np
from ultralytics import YOLO


model = YOLO('yolov8n.pt')

print("model loaded successfully")
model.export(format="onnx")
