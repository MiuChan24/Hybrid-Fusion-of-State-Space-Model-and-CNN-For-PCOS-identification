import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. SETUP
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_DIR = "./data/dataset_B/final/test"
MODEL_PATH = "densenet121_pcos_baseline.pth"
CLASSES = ['Dominant_Follicle', 'Normal', 'PCO']

# 2. TRANSFORM
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_loader = DataLoader(datasets.ImageFolder(TEST_DIR, transform=test_transform), batch_size=16, shuffle=False)

# 3. LOAD DENSENET ARCHITECTURE
print("🔬 Loading DenseNet-121 Weights...")
model = models.densenet121(weights=None)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 3)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE)
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
print("\n📊 --- DENSENET-121 CLASSIFICATION REPORT ---")
print(classification_report(all_labels, all_preds, target_names=CLASSES, zero_division=0))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', # Green for the medical standard
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Clinical Confusion Matrix: DenseNet-121')
plt.savefig('05_DenseNet_CM.png')
print("✅ Confusion Matrix saved as '05_DenseNet_CM.png'")
