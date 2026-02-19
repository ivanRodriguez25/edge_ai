from ultralytics import YOLO


#model = YOLO("yolo26n.pt")

#model.export(format="engine")


trt_model = YOLO("yolo26n.engine")

results = trt_model("https://ultralytics.com/images/bus.jpg)")
