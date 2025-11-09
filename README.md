# 📊 Lending Club Loan Prediction Project

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2.2-lightgrey.svg)](https://scikit-learn.org/)
[![Data](https://img.shields.io/badge/Dataset-Lending%20Club-green.svg)](https://www.kaggle.com/datasets)

---

## 🚀 Project Overview

This project analyzes the **Lending Club loan dataset** and builds models to **predict loan status** for borrowers.  

It demonstrates:

- **Data cleaning and preprocessing**:
  - Removing irrelevant or low-value columns
  - Handling missing values
  - Encoding categorical variables
- **Exploratory Data Analysis (EDA)** to understand distributions and relationships
- **Machine Learning models**:
  - Random Forest Classifier
  - Deep Neural Network using TensorFlow
- **Model evaluation**:
  - Confusion matrices
  - Classification reports
  - Accuracy metrics

---

## 🛠️ Project Structure

lending-club-ml/

├── Lendingclub.py # Data cleaning + EDA 
├── requirements.txt
├── README.md

### `Lendingclub.py`
- Loads `loan.csv`
- Performs **data cleaning**:
  - Drops low-value or highly null columns
  - Fills missing values
  - Converts categorical features using one-hot encoding
- Prepares `X` and `y` for ML models
- Splits dataset into training and test sets
- Trains a **Random Forest Classifier** and a **Deep Neural Network**
- Evaluates models and prints:
  - Confusion matrices
  - Classification reports
  - Accuracy scores

---

## 💻 How to Run

 **Steps**
```bash
git clone https://github.com/yourusername/lending-club-ml.git
cd lending-club-ml

# Make sure you're in the project folder
pip install -r requirements.txt

# Run the full data cleaning, model training, and evaluation
python Lendingclub.py



