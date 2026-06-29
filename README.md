# Hybrid-Fusion-of-State-Space-Model-and-CNN-For-PCOS-identification
PyTorch implementation of a hybrid State Space and CNN ensemble for denoised, multi-class ovarian morphology classification.
# Domain-Synced Soft-Voting Ensemble for Automated PCOS Diagnosis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21046275.svg)](https://doi.org/10.5281/zenodo.21046275)



## Overview
This repository contains the official code and evaluation framework for our diagnostic pipeline aimed at automating the detection of Polycystic Ovary Syndrome (PCOS) from ultrasound imagery. 

By integrating state-space models (**MedMamba**) with established Convolutional Neural Networks (**ResNet-50**), we propose a Domain-Synced Soft-Voting Ensemble. The system categorizes ultrasound frames into three distinct clinical classes: **Normal**, **PCO (Polycystic Ovaries)**, and **Dominant Follicle**.

> **Note:** The pre-trained model weights (`.pth`) and the sanitized clinical dataset are hosted externally on Zenodo to comply with Git repository file limits and maintain a lightweight codebase.

---

## 📊 1. Proposed Pipeline Architecture
![Pipeline Architecture](docs/pipeline_architecture.png) 
*(Please see the original manuscript for the high-resolution architecture diagram detailing the integration of MedMamba and ResNet-50).*

<img width="756" height="1300" alt="image" src="https://github.com/user-attachments/assets/61b34e16-931e-4f09-b095-a6bcc12210bb" />


Methodology of Project

---

## 🗂️ 2. Dataset and Preprocessing
The raw ultrasound images utilized in this study were sourced from publicly available clinical datasets (Choudhari et al. and Ashadzzaman et al.). To ensure robust model training and clinical relevance, the data underwent a strict preprocessing pipeline:
1.  **Cryptographic Sanitization:** Perceptual Hashing (pHash) was applied to detect and remove duplicated or overlapping frames.
2.  **Spatial Standardization:** Images were subjected to padded resizing to $224 \times 224$ pixels to maintain aspect ratio integrity.
3.  **Contrast Enhancement:** Contrast Limited Adaptive Histogram Equalization (CLAHE) was applied to mitigate acoustic speckle noise and enhance follicular boundary definition.

**Data Availability:** The fully processed, CLAHE-enhanced dataset is permanently archived and publicly accessible via Zenodo at: [Insert Zenodo DOI Link here].

---

## 🏗️ 3. Repository Structure

```text
├── evaluate/                 # Evaluation scripts for generating metrics and matrices
│   ├── evaluate_resnet.py
│   ├── evaluate_densenet.py
│   ├── evaluate_medmamba.py
│   └── evaluate_ensemble.py
├── scripts/                  # Scripts utilized for model training and ablation studies
├── utils/                    # Data engineering and processing tools
│   ├── apply_clahe.py
│   ├── clean_dataset.py
│   └── plot_metrics.py
├── weights/                  # Directory for local .pth weights (Ignored by Git)
├── README.md
├── requirements.txt
└── .gitignore
```
## ⚙️ 4. Installation & Setup
Clone the repository:

Bash
git clone [https://github.com/YourUsername/Hybrid-Fusion-of-State-Space-Model-and-CNN-For-PCOS-identification
.git](https://github.com/YourUsername/Hybrid-Fusion-of-State-Space-Model-and-CNN-For-PCOS-identification
.git)

cd Hybrid-Fusion-of-State-Space-Model-and-CNN-For-PCOS-identification


Install dependencies:

```Bash
pip install -r requirements.txt 


Download Weights & Data:

Download the processed dataset and pre-trained weights from [Zenodo DOI].

Place the images in a local data/ folder and the .pth files in the weights/ folder.
```


## 🚀 5. Usage
Preprocessing
To apply the pHash cleaning and CLAHE enhancement to a new raw dataset:

```bash
python utils/clean_dataset.py --input_dir data/raw --output_dir data/processed
python utils/apply_clahe.py --input_dir data/processed --output_dir data/clahe_enhanced
Evaluation
To evaluate the proposed ensemble model and generate the final Confusion Matrix and ROC curves:

```bash
python evaluate/evaluate_ensemble.py --weights_path weights/ --data_dir data/clahe_enhanced
```

## 📈 6. Results
The Domain-Synced Soft-Voting Ensemble successfully leverages the global context extraction of MedMamba alongside the localized feature extraction of ResNet-50, demonstrating superior performance over baseline models (DenseNet-121, EfficientNet) across all clinical metrics.
### Quantitative Performance Comparison

The table below details the performance of the proposed Domain-Synced Soft-Voting Ensemble against standard baseline architectures on the CLAHE-enhanced test set.

| Architecture | Macro Precision | Macro Recall | Macro F1-Score | Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Final Synced Ensemble** | **0.93** | **0.90** | **0.91** | **0.90** |
| Hybrid Ensemble (Mamba + ResNet) | 0.88 | 0.82 | 0.83 | 0.82 |
| ResNet-50 (CLAHE) | 0.85 | 0.87 | 0.86 | 0.87 |
| ResNet-50 (Standard) | 0.91 | 0.85 | 0.87 | 0.87 |
| DenseNet-121 | 0.89 | 0.83 | 0.85 | 0.84 |
| EfficientNet-B0 | 0.76 | 0.79 | 0.75 | 0.75 |
| Penalized MedMamba | 0.82 | 0.73 | 0.76 | 0.75 |


<img width="939" height="733" alt="image" src="https://github.com/user-attachments/assets/6615f298-6a91-4981-8a6f-c709216c6ecf" />

*Note: The proposed ensemble demonstrates the highest performance across all primary classification metrics, effectively balancing sensitivity and specificity for clinical diagnostic utility.*



## 🖋️ 7. Author & Citation
Neha Singh

Manipal University Jaipur


Acknowledgments: Baseline architectures were imported via standard torchvision implementations. The MedMamba backbone architecture was adapted from the original repository by Yue et al.


***

