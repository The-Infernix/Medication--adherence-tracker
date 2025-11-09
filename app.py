from ultralytics import YOLO
from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import time
import base64
from io import BytesIO

# Load YOLO model
MODEL_PATH = "C:/OBJECT DETECTION/Pill Training/runs/detect/train4/weights/best.pt"
model = YOLO(MODEL_PATH)

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Pill Detection API is running!"

@app.route('/detect', methods=['POST'])
def detect():
    start_time = time.time()
    
    # Check if image is sent
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Read image bytes
    file = request.files['image'].read()
    npimg = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Run YOLO detection
    results = model(frame, verbose=False)
    detections = results[0].boxes

    # Draw bounding boxes on the frame
    for box, score in zip(detections.xyxy, detections.conf):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Count pills detected
    pill_count = len(detections)
    detected = pill_count > 0

    # Calculate FPS
    fps = round(1.0 / (time.time() - start_time), 2)

    # Encode image to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    # Return JSON with image + info
    return jsonify({
        "pill_detected": detected,
        "pill_count": pill_count,
        "fps": fps,
        "image": jpg_as_text  # 🔥 Base64 image with boxes
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
