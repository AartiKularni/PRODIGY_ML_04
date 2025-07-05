# PRODIGY_ML_04
# 🍽️ Food Image Classification + Calorie Estimation

This project aims to develop a machine learning model that recognizes food items from images and estimates their calorie content. It helps users track their dietary intake and make informed food choices.

---

## 🚀 Overview

Using the [Food-101 dataset](https://www.kaggle.com/dansbecker/food-101), we trained a deep learning model to classify 101 types of food and mapped each class to an estimated calorie value using a custom lookup table.

---

## 📊 Features

- 🍕 **Food Image Classification** (101 classes)
- 🔥 **Calorie Estimation** per recognized item
- 🧠 **Pre-trained CNN** (EfficientNetB0 / ResNet50)
- 🌐 Optional Web App using **Streamlit**
- 📦 Calorie Mapping using **JSON/CSV**

---

## 📁 Dataset

- Source: [Food-101 Dataset on Kaggle](https://www.kaggle.com/dansbecker/food-101)
- Total Images: 101,000
- Classes: 101 food types
- Format: JPG images in train/test split

---

## 🧠 Model Architecture

- Transfer Learning using:
  - `EfficientNetB0` (or ResNet50 as alternative)
  - Input: 224×224 resized food image
  - Output: Predicted food class

---

## 🔢 Calorie Estimation

Each food class is mapped to an approximate calorie value based on public nutrition data.

```python
calorie_map = {
    'pizza': 285,
    'burger': 354,
    'spaghetti_bolognese': 220,
    ...
}
🧪 Evaluation
Accuracy: ~80%

Loss: ~0.78

Metrics: Top-1 Accuracy, Confusion Matrix


## 💻 Installation

To get started with this project, follow these steps:

### ✅ Prerequisites

Make sure you have the following installed:

- Python >= 3.7
- pip
- Git
- Virtual environment tool (optional but recommended)
