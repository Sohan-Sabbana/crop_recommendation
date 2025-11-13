# crop_recommendation


# 📌 Crop Recommendation System — ML Training Script

**File:** `crop_recommendation_final.py`

This script is a **complete end-to-end machine learning pipeline** for building a **Crop Recommendation System** using soil and environmental parameters. It automates dataset loading, preprocessing, model training, evaluation, hyperparameter tuning, visualization, and prediction.

---

##  What This Script Does

### 1. Loads Dataset Automatically

* Reads any `.csv` file placed inside the `data/` directory.
* In Google Colab, if no dataset exists, it automatically opens a file-upload prompt.
* Displays the dataset shape and first few rows.

---

### 2. Cleans and Preprocesses the Data

* Removes duplicate rows.
* Strips whitespace in column names.
* Automatically detects the **label column** (crop name).
* Handles missing values:

  * Numerical → replaced with **median**
  * Categorical → replaced with **mode**
* Selects only **numeric features** for model input:

  * N, P, K
  * temperature
  * humidity
  * pH
  * rainfall
  * moisture (if present)

---

### 3. Encodes Labels & Splits Dataset

* Uses `LabelEncoder` to convert crop names into integers.
* Performs **stratified train-test split** (80–20 split).
* Applies **StandardScaler** for models that need feature scaling (SVC, KNN).

---

### 4. Trains Multiple ML Models

The script trains and evaluates 5 different models:

* **Random Forest**
* **XGBoost**
* **LightGBM**
* **Support Vector Classifier (SVC)**
* **K-Nearest Neighbors (KNN)**

For each model it prints:

* Accuracy
* Classification report
* Confusion matrix
* ROC-AUC (if model supports probability)
* 5-fold cross-validation accuracy

---

### **5. Hyperparameter Tuning of Best Model**

* Selects the model with the **highest baseline accuracy**.
* If the model is a tree-based algorithm (RF, XGBoost, LightGBM), it performs a **small GridSearchCV** to improve performance.

---

### **6. Final Evaluation**

After tuning, the script:

* Retrains the best model
* Evaluates it on the test set again
* Prints accuracy, classification report, confusion matrix, ROC-AUC

---

### **7. Saves Model Artifacts**

Stored inside the `models/` folder:

| File                             | Purpose                                        |
| -------------------------------- | ---------------------------------------------- |
| `best_model.joblib`              | Final trained ML model                         |
| `label_encoder.joblib`           | Encoder mapping class numbers → crop names     |
| `scaler.joblib`                  | Scaler used for models requiring normalization |
| `test_classification_report.csv` | Detailed test-set metrics                      |

---

### **8. Feature Importance Visualization**

If the best model supports `feature_importances_`, the script generates:

* Top features printout
* A bar plot showing feature rankings

---

### **9. Confusion Matrix Heatmap**

Displays a heatmap of actual vs predicted crops using seaborn.

---

### **10. SHAP Explainability (Optional)**

If SHAP is installed:

* Computes SHAP values (on a sample of training data)
* Shows SHAP summary plot revealing feature impact

---

### **11. Built-in Crop Recommendation Function**

The script defines:

```python
def recommend_crop(sample_dict):
```

You can pass inputs like:

```python
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.8,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 200.0
}
```

And it returns the **predicted crop**.

The script also prints a demo prediction using median feature values.

---

## Summary of Capabilities

| Feature                           | Supported |
| --------------------------------- | --------- |
| Automatic dataset loading         | yes       |
| Data cleaning & preprocessing     | yes      |
| Multi-model training & comparison | yes       |
| Hyperparameter tuning             | yes        |
| Feature importance plots          | yes         |
| SHAP explainability               | yes         |
| Saved model for later use         | yes         |
| Prediction function included      | yes        |

---

## 🛠 Requirements

The script uses the following libraries:

* pandas
* numpy
* scikit-learn
* xgboost
* lightgbm
* shap (optional)
* seaborn
* matplotlib
* joblib

It automatically installs missing packages when run in Google Colab.

---

##  Purpose

This script is ideal for:

* Agriculture ML projects
* Smart farming applications
* Crop recommendation systems
* Academic & research work
* Demonstrating model comparison, tuning, and explainability

It provides a complete reproducible pipeline from dataset to final trained model.

