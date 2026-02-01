

#  Diabetes Prediction System

This project implements an end-to-end Machine Learning pipeline to predict whether a person has diabetes using medical measurements from the Pima Indians Diabetes Dataset. The system includes data preprocessing, model training, hyperparameter tuning, and deployment through a Gradio web interface hosted on Hugging Face Spaces.

---

## Project Overview

- **Task Type:** Binary Classification  
- **Dataset:** Pima Indians Diabetes Dataset  
- **Target Variable:** Outcome (0 = No Diabetes, 1 = Diabetes)  
- **Model Used:** Logistic Regression  
- **Evaluation Metric:** F1 Score  

The application allows users to input medical values and instantly receive a diabetes risk prediction along with confidence scores.

---

## Machine Learning Workflow

### 1. Data Preprocessing
- Replaced medically impossible zero values with NaN
- Median imputation for missing values
- Feature scaling using StandardScaler
- All preprocessing performed inside a Scikit-Learn Pipeline to prevent data leakage

### 2. Model Training
- Logistic Regression selected as the primary model due to its interpretability and suitability for binary classification problems
- 80/20 Train-Test split with stratification

### 3. Cross-Validation
- 5-fold Cross Validation using F1 score to handle class imbalance

### 4. Hyperparameter Tuning
- GridSearchCV applied on regularization parameter `C`

### 5. Final Model
- Best model selected based on highest cross-validated F1 score
- Saved using Pickle for deployment

---

## Web Application

A Gradio interface is used to collect user inputs and display:

- Diabetes prediction result
- Confidence percentage
- Class probability table

The application is deployed on Hugging Face Spaces and is publicly accessible.

---

## Input Features

- Pregnancies  
- Glucose  
- Blood Pressure  
- Skin Thickness  
- Insulin  
- BMI  
- Diabetes Pedigree Function  
- Age  

---

##  Output

- Prediction: *Diabetes Detected* / *No Diabetes Detected*
- Confidence Score
- Probability distribution for both classes

---

## How to Run Locally

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
````

### Step 2 — Run Application

```bash
python app.py
```

Open the local URL shown in the terminal.

---

## Project Structure

```
diabetes-app/
├── app.py
├── diabetes_model.pkl
├── requirements.txt
└── README.md
```

## Deployment

Hugging Face Space URL: *(add your link here)* \
GitHub Repository: *(add your link here)* \
Google Colab Notebook: *(add your link here)*

