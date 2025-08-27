from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
from fraud_detection_system import FraudDetectionSystem

app = Flask(__name__)
app.secret_key = 'fraud_detection_secret_key_2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global fraud detection system instance
fraud_system = FraudDetectionSystem()

@app.route('/')
def index():
    return render_template('fast.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Check file size first
            file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            
            if file_size > 50:  # If file is larger than 50MB
                # Use chunked reading for large files
                chunk_size = 10000  # Read 10k rows at a time
                chunks = []
                total_rows = 0
                
                for chunk in pd.read_csv(filepath, chunksize=chunk_size):
                    chunks.append(chunk)
                    total_rows += len(chunk)
                    
                    # Limit to first 100k rows for faster processing
                    if total_rows >= 100000:
                        break
                
                # Combine chunks
                data = pd.concat(chunks, ignore_index=True)
                fraud_system.data = data
                
                session['data_loaded'] = True
                session['filename'] = f"{filename} (Sample: {len(data):,} rows)"
                
            else:
                # For smaller files, load normally
                if fraud_system.load_data(filepath):
                    session['data_loaded'] = True
                    session['filename'] = filename
                else:
                    return jsonify({'error': 'Failed to load data'}), 400
            
            # Get basic data info
            data_info = {
                'shape': fraud_system.data.shape,
                'columns': list(fraud_system.data.columns),
                'fraud_percentage': 0
            }
            
            if 'Class' in fraud_system.data.columns:
                fraud_count = fraud_system.data['Class'].value_counts()
                if 1 in fraud_count:
                    data_info['fraud_percentage'] = round((fraud_count[1] / len(fraud_system.data)) * 100, 2)
            
            return jsonify({
                'success': True,
                'message': f'Data loaded successfully! Shape: {fraud_system.data.shape}',
                'data_info': data_info,
                'file_size_mb': round(file_size, 2)
            })
            
        except Exception as e:
            return jsonify({'error': f'Error loading data: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400

@app.route('/preprocess', methods=['POST'])
def preprocess_data():
    if not session.get('data_loaded'):
        return jsonify({'error': 'No data loaded. Please upload data first.'}), 400
    
    try:
        if fraud_system.preprocess_data():
            session['preprocessed'] = True
            return jsonify({
                'success': True,
                'message': 'Data preprocessed successfully!',
                'training_shape': fraud_system.X_train.shape,
                'testing_shape': fraud_system.X_test.shape
            })
        else:
            return jsonify({'error': 'Failed to preprocess data'}), 400
    except Exception as e:
        return jsonify({'error': f'Error during preprocessing: {str(e)}'}), 500

@app.route('/train', methods=['POST'])
def train_model():
    if not session.get('preprocessed'):
        return jsonify({'error': 'Data not preprocessed. Please preprocess data first.'}), 400
    
    model_type = request.json.get('model_type', 'logistic_regression')
    
    try:
        if fraud_system.train_model(model_type):
            session['trained'] = True
            session['model_type'] = model_type
            
            # Get training accuracy
            y_train_pred = fraud_system.model.predict(fraud_system.X_train_scaled)
            from sklearn.metrics import accuracy_score
            train_accuracy = accuracy_score(fraud_system.y_train, y_train_pred)
            
            return jsonify({
                'success': True,
                'message': f'{model_type.replace("_", " ").title()} model trained successfully!',
                'training_accuracy': round(train_accuracy, 4),
                'model_type': model_type
            })
        else:
            return jsonify({'error': 'Failed to train model'}), 400
    except Exception as e:
        return jsonify({'error': f'Error during training: {str(e)}'}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate_model():
    if not session.get('trained'):
        return jsonify({'error': 'Model not trained. Please train model first.'}), 400
    
    try:
        # Make predictions on test set
        y_pred = fraud_system.model.predict(fraud_system.X_test_scaled)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        test_accuracy = accuracy_score(fraud_system.y_test, y_pred)
        
        # Get confusion matrix (no plot)
        cm = confusion_matrix(fraud_system.y_test, y_pred)
        
        # Get classification report
        report = classification_report(fraud_system.y_test, y_pred, output_dict=True)
        
        return jsonify({
            'success': True,
            'test_accuracy': round(test_accuracy, 4),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        })
    except Exception as e:
        return jsonify({'error': f'Error during evaluation: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict_transaction():
    if not session.get('trained'):
        return jsonify({'error': 'Model not trained. Please train model first.'}), 400
    
    try:
        # Get transaction data from request
        transaction_data = request.json.get('transaction_data', [])
        
        if not transaction_data:
            return jsonify({'error': 'No transaction data provided'}), 400
        
        # Convert to numpy array and reshape
        transaction_array = np.array(transaction_data).reshape(1, -1)
        
        # Make prediction
        prediction = fraud_system.predict_fraud(transaction_array)
        
        return jsonify({
            'success': True,
            'prediction': prediction
        })
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

@app.route('/status')
def get_status():
    status = {
        'data_loaded': session.get('data_loaded', False),
        'preprocessed': session.get('preprocessed', False),
        'trained': session.get('trained', False),
        'model_type': session.get('model_type', None),
        'filename': session.get('filename', None)
    }
    return jsonify(status)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
