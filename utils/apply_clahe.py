import cv2
import os
import glob

# 1. SETUP PATHS
# Input paths (Your current cleaned dataset)
TRAIN_DIR_IN = "./data/dataset_B/final/train"
TEST_DIR_IN = "./data/dataset_B/final/test"

# Output paths (Where the new images will go)
TRAIN_DIR_OUT = "./data/dataset_B/clahe_final/train"
TEST_DIR_OUT = "./data/dataset_B/clahe_final/test"
CLASSES = ['Dominant_Follicle', 'Normal', 'PCO']

# 2. CREATE OUTPUT DIRECTORIES
for class_name in CLASSES:
    os.makedirs(os.path.join(TRAIN_DIR_OUT, class_name), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR_OUT, class_name), exist_ok=True)

# 3. INITIALIZE CLAHE
# clipLimit controls how "extreme" the contrast gets. 2.0 is the clinical standard.
# tileGridSize divides the image into 8x8 squares to balance contrast locally.
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def process_and_save_images(input_dir, output_dir):
    print(f"⚙️ Processing directory: {input_dir}")
    for class_name in CLASSES:
        in_class_dir = os.path.join(input_dir, class_name)
        out_class_dir = os.path.join(output_dir, class_name)
        
        # Get all image paths in this class folder
        image_paths = glob.glob(os.path.join(in_class_dir, "*.*"))
        
        for img_path in image_paths:
            # Read image in Grayscale (Ultrasounds are 1-channel anyway)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"⚠️ Warning: Could not read {img_path}. Skipping.")
                continue
            
            # Apply CLAHE
            clahe_img = clahe.apply(img)
            
            # Save the new image
            filename = os.path.basename(img_path)
            save_path = os.path.join(out_class_dir, filename)
            cv2.imwrite(save_path, clahe_img)
            
        print(f"  ✅ Processed {len(image_paths)} images for {class_name}")

# 4. EXECUTE
print("🚀 Starting CLAHE Preprocessing Pipeline...")
process_and_save_images(TRAIN_DIR_IN, TRAIN_DIR_OUT)
process_and_save_images(TEST_DIR_IN, TEST_DIR_OUT)
print("\n🎉 CLAHE Preprocessing Complete! Your new dataset is in './data/dataset_B/clahe_final/'")
