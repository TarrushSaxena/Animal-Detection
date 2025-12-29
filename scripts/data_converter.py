"""
Data Converter for Animals Detection Dataset
This dataset has images organized in class folders (e.g., lion/, tiger/)
We'll create YOLO format labels assuming each image is a full-frame detection.
"""
import os
import shutil
import random
import yaml
import kagglehub
from pathlib import Path
from PIL import Image

# Target classes we want to use (case-insensitive matching)
TARGET_CLASSES = {
    'lion': 0,
    'tiger': 1, 
    'bear': 2,
    'elephant': 3,
    'zebra': 4,
    'giraffe': 5,
    'wolf': 6,
    'leopard': 7,
    'cheetah': 8,
    'deer': 9,
}

# Directory Configuration
BASE_DIR = Path(r"C:\Users\tarus\Downloads\Animal_Detection")
DATASET_DIR = BASE_DIR / "data" / "animal_dataset"
IMAGES_TRAIN = DATASET_DIR / "images" / "train"
IMAGES_VAL = DATASET_DIR / "images" / "val"
LABELS_TRAIN = DATASET_DIR / "labels" / "train"
LABELS_VAL = DATASET_DIR / "labels" / "val"

def setup_directories():
    """Create output directories"""
    for d in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
        d.mkdir(parents=True, exist_ok=True)
    print("Output directories ready.")

def create_yolo_label(class_id, output_path):
    """
    Create a YOLO format label file.
    Since this dataset doesn't have bounding boxes, we assume full-frame detection.
    Format: class_id x_center y_center width height (normalized 0-1)
    Full frame = 0.5 0.5 1.0 1.0
    """
    with open(output_path, 'w') as f:
        # Full frame bounding box (center at 0.5, 0.5 with full width/height)
        f.write(f"{class_id} 0.5 0.5 0.9 0.9\n")

def process_dataset():
    print("=" * 50)
    print("Animals Detection Data Converter")
    print("=" * 50)
    
    # Download dataset
    print("\n[1/4] Downloading dataset...")
    try:
        path = kagglehub.dataset_download("antoreepjana/animals-detection-images-dataset")
        print(f"Dataset path: {path}")
    except Exception as e:
        print(f"Error: {e}")
        return

    setup_directories()
    
    # Find all class folders
    print("\n[2/4] Scanning for animal classes...")
    all_images = []
    
    for root, dirs, files in os.walk(path):
        # Get the folder name as potential class
        folder_name = os.path.basename(root).lower()
        
        # Check if this folder matches our target classes
        if folder_name in TARGET_CLASSES:
            class_id = TARGET_CLASSES[folder_name]
            print(f"  Found class: {folder_name} (ID: {class_id})")
            
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = Path(root) / file
                    all_images.append((img_path, class_id, folder_name))
    
    if not all_images:
        print("\nNo matching images found! Let me check what's available...")
        print("Available folders in dataset:")
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                sub_items = os.listdir(item_path) if os.path.isdir(item_path) else []
                print(f"  {item}/ ({len(sub_items)} items)")
                # Show subfolders
                for sub in sub_items[:5]:
                    sub_path = os.path.join(item_path, sub)
                    if os.path.isdir(sub_path):
                        file_count = len([f for f in os.listdir(sub_path) if f.endswith(('.jpg', '.png'))])
                        print(f"    {sub}/ ({file_count} images)")
        return

    print(f"\nTotal images found: {len(all_images)}")
    
    # Shuffle and split
    print("\n[3/4] Splitting into train/val (80/20)...")
    random.seed(42)
    random.shuffle(all_images)
    
    split_idx = int(len(all_images) * 0.8)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    print(f"  Train: {len(train_images)} images")
    print(f"  Val: {len(val_images)} images")
    
    # Copy files
    print("\n[4/4] Copying files...")
    
    for idx, (img_path, class_id, class_name) in enumerate(train_images):
        # Create unique filename
        new_name = f"{class_name}_{idx:05d}{img_path.suffix}"
        
        # Copy image
        dest_img = IMAGES_TRAIN / new_name
        shutil.copy(img_path, dest_img)
        
        # Create label
        label_name = new_name.rsplit('.', 1)[0] + '.txt'
        create_yolo_label(class_id, LABELS_TRAIN / label_name)
    
    for idx, (img_path, class_id, class_name) in enumerate(val_images):
        new_name = f"{class_name}_{idx:05d}{img_path.suffix}"
        
        dest_img = IMAGES_VAL / new_name
        shutil.copy(img_path, dest_img)
        
        label_name = new_name.rsplit('.', 1)[0] + '.txt'
        create_yolo_label(class_id, LABELS_VAL / label_name)
    
    print(f"  Copied {len(train_images)} train images")
    print(f"  Copied {len(val_images)} val images")
    
    # Create data.yaml
    print("\nCreating data.yaml...")
    class_names = [name for name, _ in sorted(TARGET_CLASSES.items(), key=lambda x: x[1])]
    
    yaml_content = {
        'path': str(DATASET_DIR.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(TARGET_CLASSES),
        'names': class_names
    }
    
    with open(DATASET_DIR / "data.yaml", 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"\n{'=' * 50}")
    print("SUCCESS! Dataset prepared for training.")
    print(f"{'=' * 50}")
    print(f"Train images: {IMAGES_TRAIN}")
    print(f"Val images: {IMAGES_VAL}")
    print(f"Config: {DATASET_DIR / 'data.yaml'}")

if __name__ == "__main__":
    process_dataset()
