# Adversarial Vulnerabilities and Defence Strategies in Deep Learning for Breast Cancer Diagnosis

**Author:** Alexander G. Rendell
**Student ID:** 2113331
**Project:** MSc Dissertation, Computer Science, Swansea University
**Submission Date:** 29th September 2025

## Project Overview
This repository contains the full code for the dissertation
*"Adversarial Vulnerabilities and Defence Strategies in Deep Learning for Breast Cancer Diagnosis"*
I train a ResNet50 model on the BreakHis dataset and evaluate its robustness against FGSM, PGD, and C&W attack

# Dataset overview
The dataset used for breast cancer images - Histopathology
https://www.kaggle.com/datasets/ambarish/breakhis

2,480 benign - Non-cancerous (Good) - 0
5,429 malignant - Cancerous (Bad) - 1

Images are stored in PNG format
Image size: 700 x 460 pixels

Benign tumors:
Adenosis (A)
Fibroadenoma (F)
Phyllodes Tumor (PT)
Tubular Adenoma (TA)

Malignant tumors:
Ductal Carcinoma (DC)
Lobular Carcinoma (LC)
Mucinous Carcinoma (MC)
Papillary Carcinoma (PC)

## Environment 
Python 3.12
PyTorch 2 or higher (GPU recomended)
see `requirements.txt` for full library list

## Know issues
C&W may crash on CPU -> use GPU if possible.
Adversarial Training may crash if ran for too long -> Use saved model states to restart ~every 20 epochs.

## Setup
1. Clone or unzip this repository.
2. `pip install -r requirements.txt`
3. Download the BreakHis dataset from the kaggle link above - place the `breakhis` folder at the root of the repository

## Running the code
The main workflow is inside Test.ipynb
Install jupyter
if you dont have it:
```bash
pip install jupyter
jupyter lab Test.ipynb
```
and:
Follow the comments on 'Test.ipynb'

## Checkpoints
Inside the `checkpoints/` folder:
- `original_model_checkpoint_0adv_140epoch.pth` - Model trained on clean images
- `pgd_model_checkpoint_50adv_140epoch.pth` - Model trained on 50% PGD-based adversarial images
- `pgd_model_checkpoint_100adv_140epoch.pth` - Model trained on 100% PGD-based adversarial images

Load any of these in `Test.ipynb` to skip re-training

## Contact
Alexander G. Rendell - 2113331@swansea.ac.uk

