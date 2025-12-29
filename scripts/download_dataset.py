import kagglehub
import os

print("Downloading dataset...")
# Download latest version
path = kagglehub.dataset_download("antoreepjana/animals-detection-images-dataset")

print("Path to dataset files:", path)

# List files in the downloaded directory to help with the converter script
print("Files in downloaded directory:")
for root, dirs, files in os.walk(path):
    level = root.replace(path, '').count(os.sep)
    indent = ' ' * 4 * (level)
    print('{}{}/'.format(indent, os.path.basename(root)))
    for f in files:
        print('{}{}'.format(indent + '    ', f))
