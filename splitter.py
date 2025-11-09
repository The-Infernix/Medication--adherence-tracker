import os
import shutil
import random

base_dir = "data"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# Create folders
for folder in [train_dir, val_dir]:
    os.makedirs(os.path.join(folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(folder, "labels"), exist_ok=True)

# Split 80% train, 20% val
image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg") or f.endswith(".png")]
random.shuffle(image_files)
split_index = int(0.8 * len(image_files))

train_files = image_files[:split_index]
val_files = image_files[split_index:]

def copy_files(file_list, dest_dir):
    for file in file_list:
        name = os.path.splitext(file)[0]
        shutil.copy(os.path.join(images_dir, file), os.path.join(dest_dir, "images", file))
        shutil.copy(os.path.join(labels_dir, f"{name}.txt"), os.path.join(dest_dir, "labels", f"{name}.txt"))

copy_files(train_files, train_dir)
copy_files(val_files, val_dir)

print("✅ Split complete! Train/Val folders created.")
