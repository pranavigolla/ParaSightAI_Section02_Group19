---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown id="MkkUJEyT8EtD"}
# **README: Malaria Parasite Life Stage Classification (ResNet Model)**

### **PROJECT OVERVIEW**

This file is the ReadMe file for running our malaria parasite life-stage
classification model using specifically the pretrained ResNet
architecture. The goal of this project is to classify segmented malaria
parasite images into life-stage categories such as Ring, Trophozoite,
Schizont, Gametocyte, and more.

All code is provided in the Jupyter notebook titled \"Section 02, Group
19 - Final Project ResNet Model.ipynb\" and is **designed to be executed
in a Google Colab file.**
:::

::: {.cell .markdown id="sxHnfDKH-Tmv"}
### **1. INSTALLATION** {#1-installation}

1.  Open the .ipynb/Jupyter notebook file in Google Colab.
2.  Upload the dataset to your Google Drive.
3.  Mount Google Drive in the Jupyter notebook using the code below:
:::

::: {.cell .code id="GNKsRXTQ-Yus"}
``` python
from google.colab import drive
drive.mount('/content/drive')
```
:::

::: {.cell .markdown id="maPqA5p__49B"}
### **2. SETTING UP THE DATASET** {#2-setting-up-the-dataset}

The folder that contained this ReadMe file also contains a folder with a
sample dataset inside for demonstration. The notebook expects the
following structure:
:::

::: {.cell .code id="jrNOOE5tAQAH"}
``` python
/data/
    /parasite_crops/        ← all segmented/cropped parasite images
    /labels.xlsx            ← contains imageName, stage, center_x, center_y
```
:::

::: {.cell .markdown id="HKhotUfvAUfm"}
### **3. RUNNING THE MODEL** {#3-running-the-model}

1.  Open the Jupyter notebook:
    Section_02_Group_19_Final_Project_ResNet_Model.ipynb
2.  Mount Google Drive:
:::

::: {.cell .code id="UGrVYte9AsvR"}
``` python
from google.colab import drive
drive.mount('/content/drive')
```
:::

::: {.cell .markdown id="9iHQnM6_AzQ7"}
1.  In the notebook, update the file paths:
:::

::: {.cell .code id="E1ffJBkVA3YV"}
``` python
IMG_DIR = "/content/drive/MyDrive/your_image_folder"
LABEL_FILE = "/content/drive/MyDrive/labels.xlsx"
```
:::

::: {.cell .markdown id="HJnC90QvA52i"}
1.  Run the notebook cells in order. The notebook performs the following
    tasks:

-   Data loading
-   Class balancing
-   Image transformations & augmentation
-   Train/val/test splitting
-   Model setup (ResNet)
-   Training
-   Loss/accuracy curve plotting
-   Test set evaluation
-   Confusion matrix
-   Sample predictions visualization
:::

::: {.cell .markdown id="GTo-5OQcBbO4"}
### **4. OBTAINING THE SAME RESULTS** {#4-obtaining-the-same-results}

If the cells are run as-is, with no modifications, the expected results
are:

-   Training accuracy curve
-   Validation accuracy curve
-   Training & validation loss curves
-   Test accuracy \~0.55
-   Classification report (precision/recall/F1 per life stage)
-   Confusion matrix heatmap
-   Sample test predictions panel
:::

::: {.cell .markdown id="kQ7IyD1TByiv"}
### **5. SAMPLE TEST DATA AND DEMO CODE** {#5-sample-test-data-and-demo-code}

A small set of cropped parasite images is located in the following
subfolder:
:::

::: {.cell .code id="BrzIXz3wB6Nz"}
``` python
/sample_data/test_samples/
```
:::

::: {.cell .markdown id="1a5scHxlCAW1"}
To test the model on this data, run the following code:
:::

::: {.cell .code id="2mykH63OCDsl"}
``` python
from PIL import Image
import torch
import torchvision.transforms as transforms

model.eval()

img = Image.open("sample_data/test_samples/example.png").convert("RGB")

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

tensor = test_transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    pred = model(tensor).argmax(dim=1).item()

print("Predicted class:", classes[pred])
```
:::

::: {.cell .markdown id="f_AGo2vJCQaW"}
### **6. EXPECTED OUTPUTS** {#6-expected-outputs}

This folder also contains a PDF (Expected_Output_Results_ResNet.pdf)
that contains these expected results:

-   Training accuracy curve
-   Validation accuracy curve
-   Loss curves
-   Confusion matrix
-   Classification report
-   Sample predictions
:::

::: {.cell .code id="YsDywIfICnUG"}
``` python
```
:::
