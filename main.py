from ultralytics import YOLO
import cv2

# Load your trained YOLO model
model = YOLO(r"runs/detect/pill_model_v112/weights/last.pt")  # <-- Update path if different

# Open the default camera (0 = primary webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

print("✅ Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLO prediction on the current frame
    results = model.predict(source=frame, conf=0.5, verbose=False)

    # Draw the detections on the frame
    annotated_frame = results[0].plot()

    # Display the result
    cv2.imshow("Pill Detection (YOLOv8)", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("🛑 Camera stopped.")
