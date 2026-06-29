import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from MedMamba import VSSM as MedMamba

# 1. SETUP
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = "./data/dataset_B/clahe_final/test" 
CLASSES = ['Dominant_Follicle', 'Normal', 'PCO']

# 2. DATA
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_loader = DataLoader(datasets.ImageFolder(TEST_DIR, transform=test_transform), batch_size=16, shuffle=False)

# 3. INITIALIZE MODELS
model_mamba = MedMamba(num_classes=3).to(DEVICE)
model_mamba.load_state_dict(torch.load("medmamba_pcos_BEST.pth", map_location=DEVICE))
model_mamba.eval()

model_resnet = models.resnet50(weights=None)
model_resnet.fc = nn.Linear(model_resnet.fc.in_features, 3)
model_resnet.load_state_dict(torch.load("resnet50_clahe_BEST.pth", map_location=DEVICE))
model_resnet = model_resnet.to(DEVICE).eval()

# 4. EVALUATE & PLOT
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        probs_mamba = F.softmax(model_mamba(images), dim=1)
        probs_resnet = F.softmax(model_resnet(images), dim=1)
        ensemble_probs = (probs_mamba + probs_resnet) / 2.0
        
        all_labels.extend(labels.cpu().numpy())
        all_probs.append(ensemble_probs.cpu().numpy())

final_labels = np.array(all_labels)
final_probs = np.concatenate(all_probs, axis=0)

# ROC for PCO class (Index 2 in your list)
y_true = (final_labels == 2).astype(int) 
y_scores = final_probs[:, 2]

fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'Ensemble ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.savefig('Ensemble_ROC_Curve.png')
print(f"✅ ROC curve saved! Processed {len(final_labels)} samples.")
