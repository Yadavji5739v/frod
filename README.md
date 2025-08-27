# 🚀 Fraud Detection System

A comprehensive machine learning system for detecting fraudulent credit card transactions.

## Features

- **Data Loading & Exploration**: Automatically loads and analyzes credit card transaction data
- **Data Preprocessing**: Handles data cleaning, scaling, and train-test splitting
- **Model Training**: Supports Random Forest and Logistic Regression algorithms
- **Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, and confusion matrix
- **Feature Importance**: Visualizes which features are most important for fraud detection
- **Real-time Prediction**: Can predict fraud for new transactions

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the System**:
   ```bash
   python fraud_detection_system.py
   ```

## What the System Does

1. **Loads** the credit card dataset from `archive/creditcard.csv`
2. **Explores** the data to understand patterns and fraud distribution
3. **Preprocesses** the data (scaling, splitting into train/test sets)
4. **Trains** a Random Forest model on the data
5. **Evaluates** the model performance with detailed metrics
6. **Shows** feature importance to understand what drives fraud detection
7. **Demonstrates** real-time prediction on sample transactions

## Dataset

The system expects a credit card fraud dataset with:
- Multiple numerical features (V1, V2, V3, etc.)
- A 'Class' column where 0 = legitimate, 1 = fraudulent

## Model Performance

- Uses Random Forest Classifier for robust fraud detection
- Handles imbalanced datasets (fraud is typically rare)
- Provides confidence scores for predictions
- Shows detailed classification metrics

## Output

- Console output with detailed analysis
- Confusion matrix visualization
- Feature importance charts
- Model accuracy and performance metrics

## Customization

You can easily modify the system to:
- Use different algorithms (Logistic Regression, XGBoost, etc.)
- Add more data preprocessing steps
- Implement cross-validation
- Save and load trained models
- Create a web interface

## Requirements

- Python 3.7+
- See `requirements.txt` for package versions

---

**Ready to detect fraud! 🕵️‍♂️**
