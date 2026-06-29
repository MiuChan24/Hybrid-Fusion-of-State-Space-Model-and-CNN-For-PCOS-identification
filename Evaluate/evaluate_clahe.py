import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from MedMamba import VSSM as MedMamba

# 1. SETUP
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = "./data/dataset_B/clahe_final/test" # Pointing to the new CLAHE test set
MODEL_PATH = "medmamba_pcos_BEST.pth"          # Assuming train.py saved it here
CLASSES = ['Dominant_Follicle', 'Normal', 'PCO']

# 2. TRANSFORM (Must match training normalization)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_loader = DataLoader(datasets.ImageFolder(TEST_DIR, transform=test_transform), batch_size=16, shuffle=False)

# 3. LOAD MEDMAMBA ARCHITECTURE
print("🔬 Loading CLAHE-Enhanced MedMamba Weights...")
model = MedMamba(num_classes=3)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(DEVICE)
model.eval()

all_preds = []
all_labels = []

# 4. INFERENCE LOOP
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 5. METRICS & VISUALIZATION
print("\n📊 --- MEDMAMBA (CLAHE ENHANCED) CLASSIFICATION REPORT ---")
print(classification_report(all_labels, all_preds, target_names=CLASSES, zero_division=0))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', # Purple for the CLAHE run
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Clinical Confusion Matrix: MedMamba + CLAHE')
plt.savefig('06_MedMamba_CLAHE_CM.png')
print("✅ Confusion Matrix saved as '06_MedMamba_CLAHE_CM.png'")
