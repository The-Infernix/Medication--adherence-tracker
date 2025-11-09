import os
import cv2

# Input dataset folders
DATASET_DIR = r"C:/OBJECT DETECTION/Pill Training/pills-dataset"
OUTPUT_DIR = r"C:/OBJECT DETECTION/Pill Training/pills-detection-yolo"

# Create output structure
for split in ["train", "valid"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

# Function to create YOLO label
def create_yolo_label(image_path, label_path):
    img = cv2.imread(image_path)
    if img is None:
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = img.shape[:2]
    with open(label_path, "w") as f:
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw * bh < 1000:  # ignore tiny noise
                continue
            # YOLO format: class x_center y_center width height (normalized)
            x_c = (x + bw / 2) / w
            y_c = (y + bh / 2) / h
            f.write(f"0 {x_c} {y_c} {bw/w} {bh/h}\n")

# Convert dataset
for split in ["train", "valid"]:
    for folder in os.listdir(os.path.join(DATASET_DIR, split)):
        folder_path = os.path.join(DATASET_DIR, split, folder)
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                src = os.path.join(folder_path, img_file)
                dst_img = os.path.join(OUTPUT_DIR, split, "images", img_file)
                dst_label = os.path.join(OUTPUT_DIR, split, "labels", os.path.splitext(img_file)[0] + ".txt")

                cv2.imwrite(dst_img, cv2.imread(src))
                create_yolo_label(src, dst_label)

print("✅ Conversion completed! YOLO dataset created at:", OUTPUT_DIR)
