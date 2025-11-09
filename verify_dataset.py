import os
import yaml
import glob

# === CONFIG ===
DATA_DIR = "data"
CLASSES_FILE = os.path.join(DATA_DIR, "classes.txt")
YAML_PATH = "data.yaml"

# === CHECK CLASSES FILE ===
if not os.path.exists(CLASSES_FILE):
    raise FileNotFoundError(f"❌ Missing classes.txt in {DATA_DIR}")

with open(CLASSES_FILE, "r") as f:
    class_names = [line.strip() for line in f if line.strip()]

print(f"✅ Found {len(class_names)} classes: {class_names}")

# === CHECK FOLDER STRUCTURE ===
required_subdirs = ["train/images", "train/labels", "val/images", "val/labels"]
for sub in required_subdirs:
    full = os.path.join(DATA_DIR, sub)
    if not os.path.exists(full):
        raise FileNotFoundError(f"❌ Missing folder: {full}")
print("✅ Folder structure OK")

# === VERIFY FILE PAIRS ===
def check_pairs(folder):
    imgs = sorted(glob.glob(os.path.join(folder, "images", "*.jpg")) + glob.glob(os.path.join(folder, "images", "*.png")))
    labels = sorted(glob.glob(os.path.join(folder, "labels", "*.txt")))

    img_basenames = set(os.path.splitext(os.path.basename(f))[0] for f in imgs)
    label_basenames = set(os.path.splitext(os.path.basename(f))[0] for f in labels)

    missing_labels = img_basenames - label_basenames
    missing_imgs = label_basenames - img_basenames

    return missing_labels, missing_imgs

missing_train_labels, missing_train_imgs = check_pairs(os.path.join(DATA_DIR, "train"))
missing_val_labels, missing_val_imgs = check_pairs(os.path.join(DATA_DIR, "val"))

if missing_train_labels or missing_val_labels:
    print(f"⚠️ Missing labels for: {missing_train_labels | missing_val_labels}")
if missing_train_imgs or missing_val_imgs:
    print(f"⚠️ Missing images for: {missing_train_imgs | missing_val_imgs}")
if not (missing_train_labels or missing_train_imgs or missing_val_labels or missing_val_imgs):
    print("✅ All image-label pairs are correctly matched")

# === CREATE data.yaml ===
yaml_dict = {
    "path": DATA_DIR,
    "train": os.path.join(DATA_DIR, "train"),
    "val": os.path.join(DATA_DIR, "val"),
    "names": {i: name for i, name in enumerate(class_names)}
}

with open(YAML_PATH, "w") as f:
    yaml.dump(yaml_dict, f, sort_keys=False)

print(f"✅ data.yaml created at: {os.path.abspath(YAML_PATH)}")

print("\n🎯 Dataset verified and ready for YOLO training!")
