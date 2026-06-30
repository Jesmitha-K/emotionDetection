import os
import random
import shutil

# -----------------------------
# Paths
# -----------------------------
dataset_path = r"C:\Users\jesmi\Downloads\emotion detection\emotion detection\dataset"

train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "val")

# Validation split
VAL_SPLIT = 0.2

random.seed(42)

# Create val folder
os.makedirs(val_path, exist_ok=True)

# Process each class
for class_name in os.listdir(train_path):

    class_train = os.path.join(train_path, class_name)

    if not os.path.isdir(class_train):
        continue

    class_val = os.path.join(val_path, class_name)
    os.makedirs(class_val, exist_ok=True)

    images = [img for img in os.listdir(class_train)
              if img.lower().endswith((".jpg", ".jpeg", ".png"))]

    random.shuffle(images)

    val_count = int(len(images) * VAL_SPLIT)

    val_images = images[:val_count]

    for img in val_images:
        shutil.move(
            os.path.join(class_train, img),
            os.path.join(class_val, img)
        )

    print(f"{class_name}: moved {val_count} images")

print("\nValidation dataset created successfully!")