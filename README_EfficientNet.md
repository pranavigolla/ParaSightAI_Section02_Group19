
# Malaria Parasite Life-Stage Classification Using EfficientNet-B0

This repository presents a deep-learning pipeline for malaria parasite life-stage classification using **EfficientNet-B0**. The model is trained on cropped parasite images generated from full microscopy slides and optimized through structured preprocessing, augmentation, dataset balancing, and fine-tuning.

---

## Project Overview

Malaria diagnosis heavily relies on manual microscopy, which is slow, expertise-dependent, and difficult to scale in low-resource regions. Automated classification of parasite life stages can support faster and more consistent decision-making.

This project implements an **EfficientNet-based image classifier** to distinguish between multiple life stages including Ring, Trophozoite, Schizont, and Gametocyte.

---

## Dataset Structure

Final processed dataset:

processed_parasites_mask/
├── images/
├── masks/
├── labels.csv


Each row in `labels.csv` contains `image_path` and `label`.

---

## Preprocessing Pipeline

- Cropped parasite patches from full-slide images  
- Removed unwanted classes (WBC, GM)  
- Resized images to **224×224**  
- Normalized using ImageNet statistics  
- Stratified train/val/test split  

---

## Data Augmentation

Applied **only to the training set**, including:

- Horizontal/vertical flips  
- Rotations  
- Color jitter  
- Random affine transforms  

Augmentation helps reduce overfitting and simulate staining / imaging variability.

---

## Model Architecture: EfficientNet-B0

EfficientNet uses compound scaling of width, depth, and resolution.  
Fine-tuning strategy:

1. Load ImageNet-pretrained EfficientNet-B0  
2. Freeze convolutional backbone  
3. Train the linear classifier head  
4. Unfreeze top layers of backbone  
5. Fine-tune the network with a lower learning rate  

This approach stabilizes training and prevents catastrophic forgetting.

---

## Training Strategy

- Loss: `CrossEntropyLoss`  
- Optimizer: `Adam`  
- Learning Rate: `1e-4` + `ReduceLROnPlateau`  
- Batch Size: 32  
- Epochs: 30–50  
- Early stopping  
- Best model saved to: `efficientnet_best.pth`  

---

## Evaluation Metrics

- Overall Accuracy  
- Per-class Precision, Recall, F1  
- Confusion Matrix  
- Training/Validation Loss Curves  
- Sample Predictions (12 visualization images):
  - Raw image
  - Predicted label
  - True label

---

## Results

EfficientNet achieved **~0.55–0.60 accuracy** on the test set.

Possible factors limiting model performance:

- Class imbalance  
- Early-stage morphological similarity  
- Limited dataset size  
- Crop errors from masks  
- Stain variability across images  

---

## Future Improvements

- Improved parasite cropping via YOLO or Mask R-CNN  
- Stain normalization techniques  
- Larger, curated dataset  
- Upgrade to EfficientNet-B2/B3 or ConvNeXt  
- Use MixUp, CutMix, or RandAugment  
- Explore self-supervised learning (SimCLR, DINO)  

---

## Installation

Install dependencies:

pip install timm pandas opencv-python scikit-learn tqdm torch torchvision
