from ultralytics import YOLO
import tflite_runtime.interpreter as tflite
import numpy as np
import cv2


model = YOLO("yolov8n.pt")
model.export(format="tflite", imgsz=640)
model.export(format="tflite", int8=True)


# Crear intérprete TFLite (CPU)
MODEL_PATH = "yolov8n.tflite"

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

DST_SIZE = (640, 640)
CAMERA_PATH = "/dev/video0"


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def build_capture_instance():
    capture = cv2.VideoCapture(CAMERA_PATH, cv2.CAP_V4L2)

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_FPS, 30)

    return capture


def display_results(frame, detections):
    """
    detections debe ser un array tipo:
    [x1, y1, x2, y2, score, class]
    Adaptar según tu modelo.
    """

    for det in detections:
        x1, y1, x2, y2, score, class_id = det

        if score < 0.5:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"ID:{int(class_id)} {score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

    cv2.imshow("Detections", frame)
    cv2.waitKey(1)


def infer(frame: np.ndarray):

    # BGR → RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize
    input_tensor = cv2.resize(frame_rgb, DST_SIZE)

    # Expand dims
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # Ajustar tipo según modelo
    if input_details[0]['dtype'] == np.float32:
        input_tensor = input_tensor.astype(np.float32) / 255.0
    else:
        input_tensor = input_tensor.astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    return output[0]


def main():

    capture = build_capture_instance()

    if not capture.isOpened():
        print("No se pudo abrir la cámara")
        return

    while True:

        ret, frame = capture.read()

        if not ret:
            continue

        prediction = infer(frame)

        # ⚠️ Debes adaptar esto al formato real de salida
        #display_results(frame, prediction)


if __name__ == "__main__":
    main()
