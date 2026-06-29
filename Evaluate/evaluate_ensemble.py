import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import MedMamba
from MedMamba import VSSM as MedMamba

# 1. SETUP & PATHS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = "./data/dataset_B/clahe_final/test" # Using the CLAHE dataset
CLASSES = ['Dominant_Follicle', 'Normal', 'PCO']

print(f"🚀 Initializing Hybrid Ensemble (MedMamba + ResNet-50) on {DEVICE}...")

# 2. DATA TRANSFORM
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_loader = DataLoader(datasets.ImageFolder(TEST_DIR, transform=test_transform), batch_size=16, shuffle=False)

# 3. LOAD MODEL A: MEDMAMBA (State Space Model)
print("🔬 Loading MedMamba Weights...")
model_mamba = MedMamba(num_classes=3)
model_mamba.load_state_dict(torch.load("medmamba_pcos_BEST.pth"))
model_mamba = model_mamba.to(DEVICE)
model_mamba.eval()

# 4. LOAD MODEL B: RESNET-50 (Convolutional Neural Network)
print("🔬 Loading ResNet-50 Weights...")
model_resnet = models.resnet50(weights=None)
num_ftrs = model_resnet.fc.in_features
model_resnet.fc = nn.Linear(num_ftrs, 3)
model_resnet.load_state_dict(torch.load("resnet50_pcos_baseline.pth"))
model_resnet = model_resnet.to(DEVICE)
model_resnet.eval()

all_preds = []
all_labels = []

# 5. ENSEMBLE INFERENCE LOOP (Soft Voting)
print("🧠 Fusing Predictions...")
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Get raw logits from both models
        logits_mamba = model_mamba(images)
        logits_resnet = model_resnet(images)
        
        # Convert logits to probabilities (0.0 to 1.0)
        probs_mamba = F.softmax(logits_mamba, dim=1)
        probs_resnet = F.softmax(logits_resnet, dim=1)
        
        # Blend the probabilities (50/50 split)
        ensemble_probs = (probs_mamba + probs_resnet) / 2.0
        
        # Get the highest agreed-upon probability
        _, predicted = ensemble_probs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 6. METRICS & VISUALIZATION
print("\n📊 --- HYBRID ENSEMBLE (MAMBA + RESNET) CLASSIFICATION REPORT ---")
print(classification_report(all_labels, all_preds, target_names=CLASSES, zero_division=0))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='magma', # Magma to symbolize the ultimate fusion
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Clinical Confusion Matrix: Hybrid Ensemble (Soft Voting)')
plt.savefig('07_Ensemble_CM.png')
print("✅ Confusion Matrix saved as '07_Ensemble_CM.png'")
