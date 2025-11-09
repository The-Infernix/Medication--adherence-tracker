from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # or yolov11n.pt if you want the newer one
model.train(data="C:\\OBJECT DETECTION\\Pill Training\\data.yaml", epochs=30, imgsz=640, name="pill_model_v8")
