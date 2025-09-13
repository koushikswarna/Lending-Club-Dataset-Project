# Loan Status Prediction

A machine learning project to predict the status of loans using historical loan data. 

---

## Table of Contents
1. [Overview](#overview)  
2. [Project Goals](#project-goals)  
3. [Dataset](#dataset)  
4. [Preprocessing](#preprocessing)  
5. [Models](#models)  
6. [Training](#training)  
7. [Results](#results)  


---

## Overview
The objective of this project is to predict loan statuses such as **Current, Default, Fully Paid, Late, In Grace Period**, or **Charged Off** using historical loan data. The project explores the effectiveness of both:

- **Random Forest Classifier** (classical machine learning)  
- **Deep Neural Network (DNN)** (neural network with multiple layers and dropout for regularization)  

---

## Project Goals
- Implement data preprocessing pipelines including missing value handling, scaling, and one-hot encoding.  
- Train and evaluate models to classify loans accurately.  
- Compare classical ML and deep learning approaches.  
- Optimize the DNN with learning rate schedules, early stopping, and batch size tuning.  

---

## Dataset
The dataset includes features related to loan applications and statuses. Target classes include:  
- `loan_status_Current`  
- `loan_status_Default`  
- `loan_status_Does not meet the credit policy. Status:Charged Off`  
- `loan_status_Does not meet the credit policy. Status:Fully Paid`  
- `loan_status_Fully Paid`  
- `loan_status_In Grace Period`  
- `loan_status_Late (16-30 days)`  
- `loan_status_Late (31-120 days)`  

---

## Preprocessing
- **Features (`X`)**: All columns excluding target columns.  
- **Target (`y`)**: One-hot encoded columns representing loan statuses.  
- **Scaling**: `StandardScaler` applied to features.  
- **Train-test split**: 50-50 split used in this project.  

> **Note:** Any print statements used during development for debugging or for any use in general have been intentionally removed from this repository for clarity.  

---

## Models

### Random Forest
- Standard `RandomForestClassifier` from scikit-learn.  
- Achieved approximately **86% test accuracy**.  

### Deep Neural Network
- **Architecture**: 128 → 64 → 8 (output), ReLU activations, dropout 0.3.  
- **Optimizer**: Adam with tuned learning rate (starting at 5e-5).  
- **Loss function**: Categorical crossentropy.  
- **Early Stopping**: Monitored validation accuracy with patience of 15 epochs.  

---

## Training
- Batch size: Typically 512, tuned based on validation performance.  
- Learning rate: Tuning required; too high led to divergence, too low led to underfitting.  
- Validation monitored using **`val_accuracy`**, ensuring the model does not overfit.  
- Early stopping used to preserve the best weights based on validation performance.  

---

## Results
| Model | Validation Accuracy | Notes |
|-------|------------------|-------|
| Random Forest | ~86% | Robust classical approach |
| Deep Neural Network | ~80% | Sensitive to learning rate; performs well with tuned hyperparameters |

> Validation accuracy is considered the primary metric due to class imbalance and multi-class nature of the target.  

---



