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

💻 Installation
bash
Copy
Edit
git clone https://github.com/yourusername/food-calorie-estimator.git
cd food-calorie-estimator
pip install -r requirements.txt
📷 How to Use
Run the model:

bash
Copy
Edit
python predict.py --image path/to/image.jpg
Output:

yaml
Copy
Edit
Prediction: Pizza 🍕
Estimated Calories: 285 kcal/slice
Streamlit App (optional):

bash
Copy
Edit
streamlit run app.py
📊 Sample Output

🛠️ Technologies Used
Python

TensorFlow / Keras

Pandas, NumPy

Streamlit (for web app)

Jupyter Notebook

Kaggle Datasets

📌 Future Enhancements
Nutritional breakdown: Protein, Carbs, Fat

Multi-item detection on a single plate

Voice and OCR input

Mobile app deployment

🧾 License
This project is open-source under the MIT License.

🙋‍♀️ Authors
Made with ❤️ by Your Name

yaml
Copy
Edit

---

Would you like me to generate this as a downloadable `.md` file or help you customize the GitHub repo structure as well?








Ask ChatGPT
