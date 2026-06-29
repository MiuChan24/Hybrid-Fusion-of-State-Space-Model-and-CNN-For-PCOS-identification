import os
from PIL import Image

# Check both train and test folders just to be perfectly safe
folders_to_check = ["./data/raw/train", "./data/raw/test"]
removed_count = 0

for folder in folders_to_check:
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Try to open and verify the file is a real image
                img = Image.open(file_path)
                img.verify() 
            except Exception:
                # If it fails, terminate the file
                print(f"🗑️ Deleting bad file: {file_path}")
                os.remove(file_path)
                removed_count += 1

print(f"\n✅ Cleanup complete. Destroyed {removed_count} corrupted/hidden files.")
