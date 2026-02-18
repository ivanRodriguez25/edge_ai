from executorch.runtime import Module
import cv2
import numpy as np
from executorch_yolo.executorch_yolo import export_yolo_executorch


def yolo_executorch():
    pass

def display_results(frame, results):

    boxes = results.boxes
    for box in boxes:
        # Convert tensors into native values
    
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        label = 'UNKNWN'

        # Draw boxes and labels
        color = (255, 0, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )

    return frame


def main():

    module = Module("yolov8n.pte")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        input_tensor = cv2.resize(frame, (640, 640))
        input_tensor = input_tensor.transpose(2, 0, 1)[None, ...]
        input_tensor = input.tensor.astype(np.float32) / 255.0

        outputs = module.execute("forward", [input_tensor])[0]

        frame = display_results(frame, outputs)
    
        cv2.imshow('Detections', frame)
        if cv2.waitKey(1) == 27:
            break


if __name__ == "__main__":

    yolo_executorch()
    main()
