import os

# Root dataset folder (update this path)
root_dir = r"C:/OBJECT DETECTION/Pill Training/dataset"

# Mapping old → new (0-indexed) YOLO class IDs
id_map = {
    15: 0,  # Norflox TZ
    16: 1,  # Meftal SPAS
    17: 2,  # Nise
    18: 3,  # Dolo 650
    19: 4   # Coldact
}

# Walk through train and val directories
for subfolder in ["train", "val"]:
    folder_path = os.path.join(root_dir, subfolder)
    if not os.path.exists(folder_path):
        continue

    for root, _, files in os.walk(folder_path):
        for file in files:
            if not file.endswith(".txt"):
                continue

            label_path = os.path.join(root, file)
            new_lines = []

            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        old_id = int(parts[0])
                        if old_id in id_map:
                            parts[0] = str(id_map[old_id])  # map old → new
                            new_lines.append(" ".join(parts))
                        else:
                            print(f"⚠️ Skipping unknown class ID {old_id} in {file}")

            # Overwrite file
            with open(label_path, "w") as f:
                f.write("\n".join(new_lines))

            print(f"✅ Fixed: {label_path}")

print("\n🎯 All label files successfully remapped to YOLO 0-based indices!")
