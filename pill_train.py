from ultralytics import YOLO
import cv2
import os
import yaml

# --- Config ---
DATASET_DIR = "dataset"
IMG_PER_PILL = 30      # Number of images to capture per pill
IMG_SIZE = 640          # YOLO input image size
CONFIDENCE = 0.5        # Detection confidence threshold
TRAINED_MODEL_PATH = r"C:/OBJECT DETECTION/Pill Training/runs/detect/pill_model/weights/best.pt"

# Create dataset directory if it doesn't exist
os.makedirs(DATASET_DIR, exist_ok=True)

# --- Menu ---
print("Select an option:")
print("1: Capture dataset for a new pill (semi-automatic labeling)")
print("2: Train YOLO model on dataset")
print("3: Use trained model for detection")
choice = input("Enter option number: ").strip()

if choice == "1":
    pill_name = input("Enter pill name: ").strip()
    pill_dir = os.path.join(DATASET_DIR, pill_name)
    os.makedirs(pill_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)  # change to 1 if your secondary camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    print(f"📸 Capturing images for pill '{pill_name}'. Press 'q' to quit early.")
    count = 0
    while count < IMG_PER_PILL:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # --- Semi-automatic bounding box (centered) ---
        box_w, box_h = int(w*0.3), int(h*0.3)  # 30% of frame
        x1 = w//2 - box_w//2
        y1 = h//2 - box_h//2
        x2 = x1 + box_w
        y2 = y1 + box_h
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow(f"Capturing {pill_name}", frame)

        # --- Save image ---
        img_path = os.path.join(pill_dir, f"{pill_name}_{count}.jpg")
        cv2.imwrite(img_path, frame)

        # --- Create YOLO label file ---
        x_center = (x1 + x2)/2 / w
        y_center = (y1 + y2)/2 / h
        box_width = box_w / w
        box_height = box_h / h
        label_path = os.path.join(pill_dir, f"{pill_name}_{count}.txt")
        with open(label_path, "w") as f:
            f.write(f"0 {x_center} {y_center} {box_width} {box_height}\n")  # class 0

        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Captured {count} images with YOLO labels for {pill_name}")

elif choice == "2":
    # --- Prepare data.yaml for YOLO ---
    class_names = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    data_yaml = {
        'path': os.path.abspath(DATASET_DIR),
        'train': 'train',
        'val': 'val',
        'names': class_names
    }
    with open("data.yaml", "w") as f:
        yaml.dump(data_yaml, f)
    print("✅ data.yaml created.")

    # --- Split images into train/val ---
    for cls in class_names:
        src_dir = os.path.join(DATASET_DIR, cls)
        train_dir = os.path.join(DATASET_DIR, 'train', cls)
        val_dir = os.path.join(DATASET_DIR, 'val', cls)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        imgs = [img for img in os.listdir(src_dir) if img.endswith(".jpg")]
        for i, img in enumerate(imgs):
            src_img = os.path.join(src_dir, img)
            dst_dir = val_dir if i % 5 == 0 else train_dir
            dst_img = os.path.join(dst_dir, img)
            if not os.path.exists(dst_img):
                os.symlink(os.path.abspath(src_img), dst_img)

    print("✅ Images organized into train/val folders.")

    # --- Train YOLO ---
    print("⏳ Training YOLO model...")
    model = YOLO("yolov8n.pt")  # pretrained YOLOv8-nano
    model.train(data="data.yaml", epochs=30, imgsz=IMG_SIZE, name="pill_model")
    print(f"✅ Training complete. Model saved as 'runs/detect/pill_model/weights/best.pt'")

elif choice == "3":
    # --- Real-time detection ---
    if not os.path.exists(TRAINED_MODEL_PATH):
        print(f"❌ Model path '{TRAINED_MODEL_PATH}' does not exist.")
        exit()

    model = YOLO(TRAINED_MODEL_PATH)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    print("✅ Detection started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        results = model.predict(source=frame, conf=CONFIDENCE, verbose=False)
        annotated_frame = results[0].plot()

        # Highlight boxes & class names
        if results[0].boxes is not None:
            for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                name = model.names[int(cls)]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(annotated_frame, name, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Pill Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Detection stopped.")

else:
    print("❌ Invalid option")
