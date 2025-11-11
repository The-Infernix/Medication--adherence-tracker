Absolutely ✅ Let’s create a **complete documentation** for your project — the **Pill Detection and Recognition System** you’ve been developing using YOLOv8 and semi-automatic dataset creation.

Here’s a well-structured, professional documentation draft 👇

---

# 🩺 **Pill Detection and Recognition using YOLOv8**

## 📘 **Overview**

This project focuses on building a **Pill Detection and Recognition System** capable of identifying different types of pills in real-time using deep learning and computer vision.
The system uses a **YOLOv8 object detection model** trained on a **custom dataset** created through semi-automatic labeling and manual refinement using tools like **LabelImg** and **Label Studio**.

---

## 🎯 **Objective**

The main goal of this project is to develop a model that can:

* Detect multiple pill types in an image or video feed.
* Assist in **automated medicine recognition** to prevent intake errors.
* Be integrated into healthcare or pharmacy automation systems.

---

## 🧠 **Model and Approach**

### **1. Dataset Preparation**

* Images of different pills were collected manually.
* Each image was labeled using **Label Studio** and **LabelImg**.
* The final dataset structure:

  ```
  data/
  ├── images/
  │   ├── train/
  │   ├── val/
  ├── labels/
  │   ├── train/
  │   ├── val/
  └── classes.txt
  ```

### **2. Dataset Classes**

Defined in `data.yaml`:

```yaml
path: data
train: train
val: val
names:
  0: Coldact
  1: Dolo 650
  2: Meftal SPAS
  3: Nise
  4: Norflox TZ
```

---

## ⚙️ **Model Training**

### **Steps:**

1. Install required dependencies:

   ```bash
   pip install ultralytics
   ```
2. Train the model using YOLOv8:

   ```bash
   yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640
   ```
3. The training results and model weights will be stored in:

   ```
   runs/detect/train/weights/best.pt
   ```

---

## 🧪 **Model Testing**

To test the trained model on new images:

```bash
yolo detect predict model=runs/detect/train/weights/best.pt source=path_to_image_or_folder
```

For example:

```bash
yolo detect predict model=best.pt source="test_images/"
```

You can also modify the detection confidence threshold:

```bash
yolo detect predict model=best.pt source="test_images/" conf=0.5
```

---

## 🖼️ **Results**

* **val_batch0_labels.jpg** → shows ground truth boxes.
* **val_batch0_pred.jpg** → shows predicted bounding boxes from the trained model.

Model achieved **strong localization accuracy** on pills like *Dolo 650* and *Coldact*, with some scope for improvement in smaller or overlapping pills.

---

## 💻 **Live Detection Script**

To test on live webcam feed or a video stream:

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

# For webcam
model.predict(source=0, show=True, conf=0.5)

# For video
# model.predict(source="pill_video.mp4", show=True)
```

---

## 📊 **Evaluation Metrics**

| Metric      | Description                                          |
| :---------- | :--------------------------------------------------- |
| Precision   | Measures model accuracy on detecting pills correctly |
| Recall      | Measures model ability to find all pills in frame    |
| mAP50       | Mean Average Precision at IoU 0.5 threshold          |
| Loss curves | Help visualize overfitting/underfitting trends       |

---

## 🚀 **Applications**

* **Pharmacy automation** for identifying pills before dispensing.
* **Elderly care support** to avoid medication confusion.
* **Medical data logging** for automatic intake tracking.
* **Healthcare AI assistants** integrated into camera-based systems.

---

## 🧩 **Future Improvements**

* Add pill **intake recognition** (detecting when a pill is taken).
* Expand dataset with **more pill types** and real-world lighting conditions.
* Integrate OCR to read **pill labels or imprints**.
* Deploy on **mobile devices** for real-time offline detection.


---

## 📁 **Repository Structure**

```
📦 PillDetection/
├── data/
├── runs/
├── detect_with_pill.py
├── data.yaml
├── requirements.txt
└── README.md
```

---

## 🧾 **References**

* [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
* [Label Studio](https://labelstud.io/)
* [LabelImg](https://github.com/heartexlabs/labelImg)

---
