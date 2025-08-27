# -*- coding: utf-8 -*-
# Fraud Detection System
# A comprehensive system for detecting fraudulent credit card transactions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionSystem:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def load_data(self, file_path):
        """Load the credit card fraud dataset"""
        try:
            self.data = pd.read_csv(file_path)
            print(f"Data loaded successfully! Shape: {self.data.shape}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def explore_data(self):
        """Basic data exploration and visualization"""
        if not hasattr(self, 'data'):
            print("No data loaded. Please load data first.")
            return
            
        print("\n=== DATA EXPLORATION ===")
        print(f"Dataset shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        print(f"\nFirst few rows:")
        print(self.data.head())
        
        # Check for missing values
        print(f"\nMissing values:\n{self.data.isnull().sum()}")
        
        # Check class distribution
        if 'Class' in self.data.columns:
            print(f"\nClass distribution:\n{self.data['Class'].value_counts()}")
            print(f"Fraud percentage: {(self.data['Class'].value_counts()[1] / len(self.data)) * 100:.2f}%")
        
        # Basic statistics
        print(f"\nBasic statistics:\n{self.data.describe()}")
    
    def preprocess_data(self):
        """Preprocess the data for training"""
        if not hasattr(self, 'data'):
            print("No data loaded. Please load data first.")
            return False
            
        print("\n=== DATA PREPROCESSING ===")
        
        # Separate features and target
        if 'Class' in self.data.columns:
            X = self.data.drop('Class', axis=1)
            y = self.data['Class']
        else:
            print("No 'Class' column found. Please check your dataset.")
            return False
        
        # Remove any non-numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_columns]
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Testing set shape: {self.X_test.shape}")
        print("Data preprocessing completed!")
        return True
    
    def train_model(self, model_type='random_forest'):
        """Train the fraud detection model"""
        if not hasattr(self, 'X_train_scaled'):
            print("Please preprocess data first.")
            return False
            
        print(f"\n=== TRAINING {model_type.upper()} MODEL ===")
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            print("Unsupported model type. Using Random Forest.")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        # Train the model
        self.model.fit(self.X_train_scaled, self.y_train)
        self.is_trained = True
        
        # Make predictions on training set
        y_train_pred = self.model.predict(self.X_train_scaled)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        
        print(f"Model trained successfully!")
        print(f"Training accuracy: {train_accuracy:.4f}")
        return True
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        if not self.is_trained:
            print("Model not trained yet. Please train the model first.")
            return
            
        print("\n=== MODEL EVALUATION ===")
        
        # Make predictions on test set
        y_pred = self.model.predict(self.X_test_scaled)
        
        # Calculate metrics
        test_accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Legitimate', 'Fraud'],
                    yticklabels=['Legitimate', 'Fraud'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return test_accuracy
    
    def predict_fraud(self, transaction_data):
        """Predict if a transaction is fraudulent"""
        if not self.is_trained:
            print("Model not trained yet. Please train the model first.")
            return None
            
        # Ensure transaction_data is in the right format
        if isinstance(transaction_data, list):
            transaction_data = np.array(transaction_data).reshape(1, -1)
        elif isinstance(transaction_data, pd.DataFrame):
            transaction_data = transaction_data.values
        
        # Scale the data
        transaction_scaled = self.scaler.transform(transaction_data)
        
        # Make prediction
        prediction = self.model.predict(transaction_scaled)
        probability = self.model.predict_proba(transaction_scaled)
        
        return {
            'prediction': 'Fraudulent' if prediction[0] == 1 else 'Legitimate',
            'confidence': max(probability[0]),
            'fraud_probability': probability[0][1]
        }
    
    def feature_importance(self):
        """Show feature importance if using Random Forest"""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            print("Feature importance not available for this model type.")
            return
            
        print("\n=== FEATURE IMPORTANCE ===")
        
        # Get feature names
        feature_names = self.X_train.columns
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 10 most important features:")
        print(importance_df.head(10))
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run the fraud detection system"""
    print("🚀 FRAUD DETECTION SYSTEM 🚀")
    print("=" * 50)
    
    # Initialize the system
    fraud_system = FraudDetectionSystem()
    
    # Load data
    print("Loading data...")
    if fraud_system.load_data('archive/creditcard.csv'):
        # Explore data
        fraud_system.explore_data()
        
        # Preprocess data
        if fraud_system.preprocess_data():
            # Train model
            fraud_system.train_model('random_forest')
            
            # Evaluate model
            fraud_system.evaluate_model()
            
            # Show feature importance
            fraud_system.feature_importance()
            
            # Example prediction
            print("\n=== EXAMPLE PREDICTION ===")
            # Get a sample transaction from test set
            sample_transaction = fraud_system.X_test.iloc[0:1]
            prediction = fraud_system.predict_fraud(sample_transaction)
            print(f"Sample transaction prediction: {prediction}")
            
            print("\n✅ Fraud Detection System is ready!")
        else:
            print("❌ Failed to preprocess data")
    else:
        print("❌ Failed to load data")

if __name__ == "__main__":
    main()
