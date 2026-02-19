from ultralytics import YOLO
import cv2
import numpy as np
import time



dst_resolution = (512, 512)

# Camera
camera_config = {
    'frame_width': dst_resolution[0], 
    'frame_height': dst_resolution[1], 
    'fps': 120, 
}

CAMERA_PATH = '/dev/video0'
capture = cv2.VideoCapture(CAMERA_PATH, cv2.CAP_V4L2)

# Define camera properties
EXPOSURE = 100
capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
capture.set(cv2.CAP_PROP_FRAME_WIDTH, dst_resolution[0])
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, dst_resolution[1])
capture.set(cv2.CAP_PROP_FPS, 120)
capture.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)

# Model
model = YOLO("yolo26n.engine", task="detect")


def display_results(frame, results, fps_):

    boxes = results.boxes
    for box in boxes:
        # Convert tensors into native values
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        label = model.names[class_id]

        # Draw boxes and labels
        color = (255, 0, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )
    cv2.putText(
        frame, f"fps: {fps_}", (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
    )

    return frame


def handle_times(dt, times):
    
    times.append(int(1/dt))

    return times[-50:]

def main():
    
    t0 = time.time()
    t1 = time.time()
    times = []

    while True:

        t1 = time.time()
        dt = t1 - t0
        t0 = time.time()
        times = handle_times(dt, times)

        fps_ = int(np.mean(times))
        
        ret, frame = capture.read()

        if not ret:
            break
        
        results = model.predict(frame)[0]
        
        frame = display_results(frame, results, fps_)
        
        cv2.imshow("Tensor RT", frame)

        if cv2.waitKey(1) == 27:
            break


if __name__ == "__main__":
    main()
