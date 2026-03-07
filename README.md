# ASD Facial Image Classification for Autism Screening

This repository contains experimental code related to the research paper:

**Taneera, S. N., Samadi, S., Özmen, S., & Alhajj, R. (2025).**  
*Game-Based Diagnosing of Children with Autism Spectrum Disorder.*  
Journal of Information & Knowledge Management.

The project explores the use of **computer vision and machine learning techniques** to support early screening of Autism Spectrum Disorder (ASD) using facial image classification.

---

## Overview

Autism Spectrum Disorder (ASD) diagnosis typically involves time-consuming behavioral and clinical evaluations.  
This project investigates whether **machine learning and deep learning models** can assist in early screening by analyzing facial images of children.

The experiments compare several deep learning architectures and classical machine learning approaches using a publicly available dataset.

---

## Methods

Two main types of approaches were explored in the experiments.

### Deep Learning (Transfer Learning)

Several convolutional neural network architectures were evaluated using transfer learning:

- EfficientNet-B3  
- ResNet50  
- MobileNet  
- Xception  

Pretrained ImageNet weights were used and the models were fine-tuned for ASD image classification.

### Classical Machine Learning

Additional exploratory experiments were conducted using traditional machine learning methods, including:

- Support Vector Machines (SVM)  
- Image preprocessing and feature flattening

These experiments were used to compare classical machine learning methods with deep learning approaches.

---

## Results

The best performing deep learning model was **EfficientNet-B3**, achieving approximately:

**Test Accuracy: 87.5%**

Additional evaluation included:

- Cross-validation experiments  
- Confusion matrix analysis  
- Precision, recall, and F1-score metrics  

For full experimental details, please refer to the published research paper.

---

## Dataset

The experiments were conducted using the:

**Autistic Children Facial Image Dataset**

Originally published on Kaggle by:

**Gerry (2020)**

The dataset is **not included in this repository** due to privacy considerations and because it is no longer publicly available at the time of this repository release.

Researchers interested in reproducing the experiments may use similar publicly available ASD facial datasets.

---

## Implementation Notes

The experimental workflow in this repository builds upon the **reference implementation and dataset originally provided by Gerry (2020)**.

Additional experiments were conducted to evaluate multiple machine learning and deep learning models, including EfficientNet-B3, ResNet50, MobileNet, and Xception, as described in the associated research paper.

This repository aims to provide transparency and documentation for the experimental setup used in the research.

---

## Acknowledgements

The original dataset and reference implementation were created by:

**Gerry (2020)** – *Autistic Children Facial Image Dataset*, Kaggle.

This work builds upon that dataset and reference implementation by conducting additional experiments and model evaluations.

The deep learning models were implemented using open-source frameworks including:

- TensorFlow  
- Keras  

---
## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```
2. Update the dataset path inside the experiment scripts

DATASET_DIR = "path_to_dataset_directory"

3.Run one of the scripts

```bash
python notebooks/compare_cnn_models.py
```
you can also run
```bash
python notebooks/cross_validation_efficientnet.py
python notebooks/classical_ml_and_cnn_experiments.py
python notebooks/adapted_cnn_experiments.py
```
## Repository Structure

- `notebooks/adapted_cnn_experiments.py`  
  Cleaned single-model transfer learning experiment.

- `notebooks/compare_cnn_models.py`  
  Comparison of Xception, MobileNetV2, ResNet50, and EfficientNetB3.

- `notebooks/cross_validation_efficientnet.py`  
  5-fold cross-validation using EfficientNetB3.

- `notebooks/classical_ml_and_cnn_experiments.py`  
  Classical machine learning and CNN-based experiments.

- `DATASET_NOTICE.md`  
  Explains why the original dataset is not included.

- `outputs/`  
  Stores generated plots, models, and result tables.
  
## Citation

If you use this repository in your research, please cite:

Taneera, S. N., Samadi, S., Özmen, S., & Alhajj, R. (2025).  
*Game-Based Diagnosing of Children with Autism Spectrum Disorder.*  
Journal of Information & Knowledge Management.  
DOI: 10.1142/S0219649225500479

## License

This repository is provided for academic and research purposes.
