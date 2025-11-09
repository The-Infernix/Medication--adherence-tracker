from ultralytics import YOLO
import cv2
import os

# 🔹 Load your trained model
model = YOLO(r"runs/detect/pill_model_v112/weights/best.pt")  # <-- change path if needed

# 🔹 Folder containing your test images
image_folder = r"C:\New Dataset"  # <-- put your test images here

# 🔹 Folder to save output images
output_folder = r"C:\OBJECT DETECTION\Pill Training\test_results"
os.makedirs(output_folder, exist_ok=True)

# 🔹 Iterate through all images in folder
for img_name in os.listdir(image_folder):
    if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(image_folder, img_name)

        # Run detection
        results = model.predict(source=img_path, conf=0.5, save=False, verbose=False)

        # Draw bounding boxes
        annotated_frame = results[0].plot()

        # Show each image (optional)
        cv2.imshow("Result", annotated_frame)
        cv2.waitKey(500)  # 0 = wait until key press, 500 = show for 0.5 sec

        # Save annotated image
        out_path = os.path.join(output_folder, f"det_{img_name}")
        cv2.imwrite(out_path, annotated_frame)
        print(f"✅ Processed: {img_name} → Saved to {out_path}")

cv2.destroyAllWindows()
print("🎯 All test images processed and saved!")
