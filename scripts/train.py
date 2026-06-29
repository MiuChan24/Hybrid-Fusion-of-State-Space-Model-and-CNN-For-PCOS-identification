import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from MedMamba import VSSM as MedMamba

# 1. PATHS & MAX PENALTY HYPERPARAMETERS
TRAIN_DIR = "./data/dataset_B/clahe_final/train"
TEST_DIR = "./data/dataset_B/clahe_final/test"
BATCH_SIZE = 16
EPOCHS = 100                  # Increased for the Scheduler
LEARNING_RATE = 0.0001
L1_LAMBDA = 2e-5              # Doubled L1 (Sparsity)
WEIGHT_DECAY = 0.1            # Doubled L2 (Weight Decay)
LABEL_SMOOTHING = 0.1         # Prevents Model Overconfidence
PATIENCE = 10                 # Early Stopping threshold
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Initializing Max-Penalty Training on: {DEVICE}")

# 2. HEAVY DATA AUGMENTATION (The 'Noise' Penalty)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(200), 
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),         # Increased rotation
    transforms.RandomGrayscale(p=0.2),     # Forces shape-learning over color/texture
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

# 4. INITIALIZE MEDMAMBA (3-CLASS)
model = MedMamba(num_classes=3) 
model = model.to(DEVICE)

# 5. OPTIMIZER, LOSS (WITH SMOOTHING), & SCHEDULER
weights = torch.tensor([0.8, 2.5, 1.0]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# 6. EARLY STOPPING TRACKERS
best_acc = 0.0
epochs_no_improve = 0

# 7. MAIN TRAINING LOOP
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Base Loss + Label Smoothing
        base_loss = criterion(outputs, labels)
        
        # Manual L1 Regularization
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = base_loss + (L1_LAMBDA * l1_norm)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
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
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] | LR: {current_lr:.6f} | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    
    # 9. EARLY STOPPING LOGIC
    if test_acc > best_acc:
        best_acc = test_acc
        epochs_no_improve = 0
        # Only save the model if it actually beat the previous best score
        torch.save(model.state_dict(), "medmamba_pcos_BEST.pth")
        print(f"🌟 New Best Model Saved with Acc: {best_acc:.2f}%")
    else:
        epochs_no_improve += 1
        print(f"⚠️ No improvement for {epochs_no_improve} epoch(s).")
        
    if epochs_no_improve >= PATIENCE:
        print(f"🛑 Early stopping triggered at Epoch {epoch+1}! Model has stopped generalizing.")
        break  # Kills the training loop

print(f"✅ Training Complete. Best model locked in at {best_acc:.2f}% Test Accuracy.")
