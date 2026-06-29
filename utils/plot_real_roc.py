import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# 1. Load your REAL data from the evaluation script
y_true = np.load('results_labels.npy')
y_probs = np.load('results_probs.npy')

# 2. Binarize the labels for multi-class ROC calculation (Classes 0, 1, 2)
y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
n_classes = y_true_bin.shape[1]

# 3. Define clinical names and professional colors matching your report
class_names = {0: 'Normal', 1: 'PCO', 2: 'Dominant Follicle'}
colors = {0: '#27ae60', 1: '#e74c3c', 2: '#2980b9'}

plt.figure(figsize=(10, 7))

# 4. Calculate and plot the real ROC curve for EACH class
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], linewidth=2.5, 
             label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

# 5. Plot the baseline (random guessing)
plt.plot([0, 1], [0, 1], color='black', linewidth=2, linestyle='--')

# 6. Add all the necessary academic formatting
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
plt.title('Multi-Class ROC Curve Analysis - Domain-Synced Ensemble', fontsize=14, fontweight='bold', pad=15)

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.05])
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc="lower right", fontsize=12)

# 7. Save the final image
plt.savefig('Figure_5_19_ROC_Curve_REAL.png', dpi=300, bbox_inches='tight')
print("Success! Your real Multi-Class ROC Curve has been saved as 'Figure_5_19_ROC_Curve_REAL.png'.")
