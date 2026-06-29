import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# 1. PATHS & HYPERPARAMETERS
TRAIN_DIR = "./data/dataset_B/final/train"
TEST_DIR = "./data/dataset_B/final/test"
BATCH_SIZE = 16
EPOCHS = 25              # Baselines converge faster
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"⚡ Initializing EfficientNet-B0 Baseline on: {DEVICE}")

# 2. STANDARD CLINICAL AUGMENTATIONS
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(200),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. LOAD DATASETS
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 4. LOAD PRE-TRAINED EFFICIENTNET-B0
# Using standard ImageNet weights for transfer learning
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

# 5. MODIFY THE HEAD FOR 3 CLASSES
# EfficientNet's classifier is a Sequential block; we replace the final linear layer
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 3) 
model = model.to(DEVICE)

# 6. OPTIMIZER & WEIGHTED LOSS (Protecting the Normal Class)
weights = torch.tensor([0.8, 2.5, 1.0]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# 7. TRAINING LOOP
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    train_acc = 100. * correct / total
    
    # 8. VALIDATION LOOP
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
    test_acc = 100. * test_correct / test_total
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    
    # Save the best EfficientNet model
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "efficientnet_pcos_baseline.pth")

print(f"✅ EfficientNet Training Complete. Best Test Accuracy: {best_acc:.2f}%")
