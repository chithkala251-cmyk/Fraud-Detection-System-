# 💳 Credit Card Fraud Detection System

A machine learning project to detect fraudulent credit card transactions using anomaly detection and classification models. Built as part of the Codec Technologies Data Science Internship.

---

## 🎯 Objective

Identify fraudulent financial transactions from highly imbalanced credit card data, so that banks and financial institutions can prevent fraud in real time — reducing financial losses and improving customer trust.

---

## 📁 Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **File:** `creditcard.csv`
- **Size:** 284,807 transactions × 31 columns
- **Features:** V1–V28 (PCA-transformed), Amount, Time
- **Target:** `Class` (0 = Legitimate, 1 = Fraud)
- **Fraud Rate:** Only 0.17% of transactions are fraudulent — extreme class imbalance

---

## 🔧 Project Pipeline

| Stage | Description |
|-------|-------------|
| **Stage 1** | Data Loading & Overview |
| **Stage 2** | Exploratory Data Analysis (EDA) |
| **Stage 3** | Data Preprocessing & SMOTE |
| **Stage 4** | Model Building & Training |
| **Stage 5** | Model Evaluation |
| **Stage 6** | Feature Importance |
| **Stage 7** | Conclusion & Business Impact |

---

## 🛠️ Technologies Used

- **Language:** Python 3
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, xgboost
- **Environment:** Jupyter Notebook

---

## 🤖 Models Used

| Model | Type |
|-------|------|
| Isolation Forest | Unsupervised Anomaly Detection |
| Random Forest Classifier | Supervised Classification |
| XGBoost Classifier | Supervised Classification |

All supervised models trained on SMOTE-balanced training data with Amount and Time scaled using StandardScaler.

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | **0.9994** | **0.8795** | 0.7684 | **0.8202** | **0.9690** |
| XGBoost | 0.9994 | 0.8352 | **0.8000** | 0.8172 | 0.9660 |
| Isolation Forest | 0.9983 | 0.0000 | 0.0000 | 0.0000 | 0.8071 |

### 🏆 Best Model: Random Forest

Random Forest achieved the highest **F1-Score (0.8202)** and **ROC-AUC (0.9690)**, correctly identifying **76.8% of all fraudulent transactions** with **87.9% precision**.

> ⚠️ Isolation Forest achieved 99.8% accuracy but F1 = 0.0 — proving that accuracy alone is a misleading metric for imbalanced fraud datasets. F1-Score and ROC-AUC are the correct evaluation metrics here.

---

## 🔍 Key EDA Findings

- Fraudulent transactions represent only **0.17%** of all transactions
- PCA-transformed features **V14, V12, V10, V17** show the strongest correlation with fraud
- Transaction amount distributions differ significantly between fraud and legitimate transactions
- Extreme class imbalance was handled using **SMOTE** on training data only

---

## ⚙️ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/fraud-detection-system.git
   cd fraud-detection-system
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
   ```

3. Place the dataset file `creditcard.csv` in the project folder

4. Open and run the notebook:
   ```bash
   jupyter notebook Fraud_Detection_System.ipynb
   ```

---

## 📌 Business Impact

An effective fraud detection system can:
- Reduce financial losses by catching fraud before it is processed
- Improve customer trust and satisfaction
- Detect fraudulent transactions in real time
- Reduce manual investigation effort significantly
- Increase overall operational efficiency
