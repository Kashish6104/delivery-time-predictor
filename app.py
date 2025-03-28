from flask import Flask, request, jsonify
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Route for predicting delivery time
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.json
    
    # Extract features from JSON data
    features = [
        data['product_category'],
        data['customer_location'],
        data['shipping_method']
    ]
    
    # Convert features to a DataFrame with One-Hot Encoding
    feature_df = pd.DataFrame([features], columns=['product_category', 'customer_location', 'shipping_method'])
    feature_df_encoded = pd.get_dummies(feature_df)
    
    # Align the columns with the model's training columns
    model_columns = X.columns  # X is the data you used to train the model earlier
    feature_df_encoded = feature_df_encoded.reindex(columns=model_columns, fill_value=0)
    
    # Make a prediction
    prediction = model.predict(feature_df_encoded)
    
    # Return the prediction as a JSON response
    return jsonify({'predicted_delivery_time': prediction[0]})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
