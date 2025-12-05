# Malaria Parasite Detection – Final Project

This repository contains the code and workflow for preparing and training deep-learning models to detect and classify malaria parasites from microscopy images. The Jupyter notebook `aiml_finalproject.ipynb` performs dataset preprocessing, YOLOv8 training, and generation of cropped parasite images for downstream classification models such as ResNet.

---

## **Project Overview**
This project aims to build a detection and classification pipeline for malaria parasite life-stages using blood smear images. The workflow includes:
- Converting microscopy masks to YOLO-formatted bounding boxes
- Generating label `.txt` files for each image
- Splitting data into **train / validation / test** folders
- Training a YOLOv8-small model for parasite detection
- Cropping YOLO-detected parasites into individual images
- Preparing these crops for a ResNet-based classification stage

---

## **Notebook Summary: What the Code Does**

### **1. Install Dependencies**
The notebook installs required packages such as OpenCV.

### **2. Generate YOLO Annotations**
- Loads original images and mask images
- Extracts connected-component contours
- Computes bounding boxes
- Converts them into YOLO’s normalized `(class, x_center, y_center, w, h)` format
- Saves each annotation as `.txt` files

### **3. Create YOLO Folder Structure**
The notebook automatically creates a directory structure with image and label splits and performs train/val/test splitting.

### **4. Build `data.yaml` for YOLOv8**
The notebook programmatically generates the configuration YAML file required by Ultralytics YOLO.

### **5. Train YOLOv8-S Model**
A pretrained YOLOv8-small model is fine-tuned on the dataset using settings such as:
- Image size: 640
- Batch size: 16
- Epochs: 10 (reduced due to runtime disconnections)

### **6. Crop YOLO-Detected Parasites for ResNet Classification**
- Loads YOLO predictions
- Extracts bounding boxes
- Crops parasite regions
- Saves class-wise cropped images for downstream ResNet training

---

## **Folder Structure**
```
project/
│── aiml_finalproject.ipynb
│── raw_images/
│── labels/
│── yolo_split/
│── cropped_for_resnet/
│── README.md
```

---

## **Requirements**
Install dependencies:
```
pip install ultralytics opencv-python-headless numpy pandas pillow scikit-learn matplotlib
```

---

## **How to Run the Notebook**
1. Open the notebook:
```
jupyter notebook aiml_finalproject.ipynb
```
2. Update dataset paths inside the notebook.
3. Run all cells sequentially.
4. YOLO outputs will appear under `runs/`.
5. Cropped images for ResNet classification will appear under `cropped_for_resnet/`.

---

## **Output**
The notebook produces:
- YOLO-formatted dataset
- Trained YOLOv8 detection model
- Cropped parasite images
- Data ready for ResNet classification

---

## **Notes**
- The notebook was originally intended to run for 100 epochs, but runtime disconnections reduced the number of completed epochs.
- Retraining for higher epochs may improve performance.

