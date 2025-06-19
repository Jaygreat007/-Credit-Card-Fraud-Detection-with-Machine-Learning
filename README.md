# 🛡️ Credit Card Fraud Detection with Machine Learning

This project detects fraudulent credit card transactions using a Random Forest Classifier. It handles a highly imbalanced dataset and demonstrates an end-to-end machine learning pipeline — from preprocessing to evaluation.

## 🔍 Project Features

- Preprocessed credit card transaction data
- Addressed class imbalance using **undersampling**
- Trained a **Random Forest Classifier**
- Evaluated using confusion matrix, precision, recall, and F1-score
- Achieved **94% accuracy** with high fraud precision
- Built using Python, pandas, scikit-learn

## 📊 Results

| Metric     | Class 0 | Class 1 (Fraud) |
|------------|---------|-----------------|
| Precision  | 0.89    | 0.98            |
| Recall     | 0.99    | 0.86            |
| F1-Score   | 0.94    | 0.92            |

## 📁 Project Files

- `test_dataset.py` - Training and evaluation script
- `fraud_detector.pkl` - Trained model (for future deployment)
- `README.md` - Project overview

## 🚀 Future Enhancements

- Build a live Streamlit web app
- Deploy the model for real-time fraud screening
- Add sample CSV for quick testing

---

> Developed by [@Jaygreat007](https://github.com/Jaygreat007)
