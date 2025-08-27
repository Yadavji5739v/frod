# Demo script for Fraud Detection System
# Shows how to use the system step by step

from fraud_detection_system import FraudDetectionSystem
import pandas as pd

def interactive_demo():
    """Interactive demo of the fraud detection system"""
    print("🎯 INTERACTIVE FRAUD DETECTION DEMO 🎯")
    print("=" * 50)
    
    # Initialize the system
    fraud_system = FraudDetectionSystem()
    
    # Step 1: Load data
    print("\n📊 Step 1: Loading Data")
    print("-" * 30)
    success = fraud_system.load_data('archive/creditcard.csv')
    
    if not success:
        print("❌ Could not load data. Please check if 'archive/creditcard.csv' exists.")
        return
    
    # Step 2: Explore data
    print("\n🔍 Step 2: Data Exploration")
    print("-" * 30)
    fraud_system.explore_data()
    
    # Step 3: Preprocess data
    print("\n⚙️ Step 3: Data Preprocessing")
    print("-" * 30)
    if not fraud_system.preprocess_data():
        print("❌ Data preprocessing failed.")
        return
    
    # Step 4: Train model
    print("\n🤖 Step 4: Model Training")
    print("-" * 30)
    fraud_system.train_model('random_forest')
    
    # Step 5: Evaluate model
    print("\n📈 Step 5: Model Evaluation")
    print("-" * 30)
    accuracy = fraud_system.evaluate_model()
    
    # Step 6: Feature importance
    print("\n🎯 Step 6: Feature Importance")
    print("-" * 30)
    fraud_system.feature_importance()
    
    # Step 7: Make predictions
    print("\n🔮 Step 7: Making Predictions")
    print("-" * 30)
    
    # Get a few sample transactions
    sample_transactions = fraud_system.X_test.iloc[0:3]
    
    for i, transaction in enumerate(sample_transactions.iterrows()):
        print(f"\nTransaction {i+1}:")
        prediction = fraud_system.predict_fraud(transaction[1])
        print(f"  Prediction: {prediction['prediction']}")
        print(f"  Confidence: {prediction['confidence']:.4f}")
        print(f"  Fraud Probability: {prediction['fraud_probability']:.4f}")
    
    print("\n🎉 Demo completed successfully!")
    print("You can now use the system to detect fraud in new transactions!")

def quick_test():
    """Quick test with minimal output"""
    print("⚡ QUICK TEST MODE ⚡")
    
    fraud_system = FraudDetectionSystem()
    
    if fraud_system.load_data('archive/creditcard.csv'):
        fraud_system.preprocess_data()
        fraud_system.train_model()
        accuracy = fraud_system.evaluate_model()
        print(f"\n✅ Quick test completed! Model accuracy: {accuracy:.4f}")
    else:
        print("❌ Quick test failed - could not load data")

if __name__ == "__main__":
    print("Choose your demo mode:")
    print("1. Interactive Demo (full experience)")
    print("2. Quick Test (minimal output)")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        interactive_demo()
    elif choice == "2":
        quick_test()
    else:
        print("Invalid choice. Running interactive demo...")
        interactive_demo()
