import torch
from executorch.exir import to_edge


def export_yolo_executorch():

    # Export YOLO
    model = torch.hub.load('ultralytics/yolov8', 'yolov8n', pretrained=True)
    model.eval()

    example = torch.rand(1, 3, 640, 640)
    edge = to_edge(model, (example, ))
    exec_prog = edge.to_executorch()

    with open("yolov8n.pte", "wb") as f:
        f.write(exec_prog.buffer)
