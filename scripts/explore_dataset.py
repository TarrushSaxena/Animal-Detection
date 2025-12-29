"""
Diagnostic script to explore the Kaggle dataset structure.
Run this and share the output so we can fix the data converter.
"""
import os
import kagglehub

# Download/get path
path = kagglehub.dataset_download("antoreepjana/animals-detection-images-dataset")
print(f"\n=== Dataset Location ===")
print(f"Path: {path}")

print(f"\n=== Top-Level Contents ===")
for item in os.listdir(path):
    full_path = os.path.join(path, item)
    if os.path.isdir(full_path):
        print(f"[DIR]  {item}")
        # Show first level inside each directory
        try:
            sub_items = os.listdir(full_path)[:10]  # First 10
            for sub in sub_items:
                sub_full = os.path.join(full_path, sub)
                if os.path.isdir(sub_full):
                    print(f"       [DIR]  {sub}")
                else:
                    print(f"       [FILE] {sub}")
            if len(os.listdir(full_path)) > 10:
                print(f"       ... and {len(os.listdir(full_path)) - 10} more items")
        except:
            pass
    else:
        print(f"[FILE] {item}")

# Look for annotation files
print(f"\n=== Looking for Annotation Files ===")
for root, dirs, files in os.walk(path):
    for f in files:
        if f.endswith(('.xml', '.csv', '.json', '.txt')):
            print(f"Found: {os.path.join(root, f)}")
            break  # Just show first one per folder
    break  # Only top level for now
