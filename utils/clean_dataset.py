import os
import shutil
import imagehash
from PIL import Image

# 1. Define safe paths
input_base = "./data/dataset_B/raw"
output_base = "./data/dataset_B/processed"
classes = ["Dominant_Follicle", "Normal", "PCO"]

print("🛡️ Initializing Non-Destructive Data Purge...")

# 2. Reset the processed directory to prevent mixing old runs
if os.path.exists(output_base):
    print(f"🧹 Clearing old processed data at {output_base}...")
    shutil.rmtree(output_base)

# 3. Process both classes automatically
for cls in classes:
    os.makedirs(os.path.join(output_base, cls), exist_ok=True)
    
    folder_path = os.path.join(input_base, cls)
    output_folder = os.path.join(output_base, cls)
    
    hash_dictionary = {}
    clones_found = 0
    unique_saved = 0
    
    print(f"\n🔍 Scanning '{cls}' folder for clones...")
    
    if not os.path.exists(folder_path):
        print(f"⚠️ Could not find {folder_path}. Skipping.")
        continue

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            filepath = os.path.join(folder_path, filename)
            
            try:
                # Calculate the structural fingerprint
                img = Image.open(filepath)
                img_hash = imagehash.phash(img)
                
                # Check for structural twins
                is_clone = False
                for saved_hash in hash_dictionary:
                    difference = img_hash - saved_hash
                    if difference < 8: # 8 is the sensitivity threshold for rotated/flipped clones
                        is_clone = True
                        break
                        
                if is_clone:
                    clones_found += 1
                    # We do NOT delete the original file, we just ignore it
                else:
                    # It's unique! Copy it to the safe 'processed' folder
                    hash_dictionary[img_hash] = filename
                    shutil.copy(filepath, os.path.join(output_folder, filename))
                    unique_saved += 1
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"  -> Ignored {clones_found} artificial clones.")
    print(f"  -> Safely copied {unique_saved} PURE images to {output_folder}")

print("\n✅ Non-Destructive Purge Complete! Your raw data remains untouched.")
