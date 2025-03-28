from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# File paths
model_path = 'model.pkl'
model_columns_path = 'model_columns.pkl'

# Load the trained model
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    model = None  # Set model to None if file is not found

# Load the model's training columns if saved separately
if os.path.exists(model_columns_path):
    with open(model_columns_path, 'rb') as f:
        model_columns = pickle.load(f)
else:
    model_columns = None  # Set model_columns to None if file is not found

# Route for predicting delivery time
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure model and model columns are loaded
        if model is None or model_columns is None:
            return jsonify({'error': 'Model or model columns file not found. Please ensure both model.pkl and model_columns.pkl are available.'}), 500
        
        # Get JSON data from the request
        data = request.json
        
        # Check if required fields are present
        required_fields = ['product_category', 'customer_location', 'shipping_method']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # Extract features from JSON data
        features = [
            data['product_category'],
            data['customer_location'],
            data['shipping_method']
        ]
        
        # Convert features to a DataFrame with One-Hot Encoding
        feature_df = pd.DataFrame([features], columns=required_fields)
        feature_df_encoded = pd.get_dummies(feature_df)
        
        # Align the columns with the model's training columns
        feature_df_encoded = feature_df_encoded.reindex(columns=model_columns, fill_value=0)
        
        # Make a prediction
        prediction = model.predict(feature_df_encoded)
        
        # Return the prediction as a JSON response
        return jsonify({'predicted_delivery_time': float(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
