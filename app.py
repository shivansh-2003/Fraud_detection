from flask import Flask, request, jsonify, make_response
from flask_talisman import Talisman
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import joblib
import logging

app = Flask(__name__)
Talisman(app)

# Setup CORS: Allow all origins with specific methods and headers
cors = CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST"], "allow_headers": ["Content-Type"]}})

# Enable logging
logging.basicConfig(level=logging.INFO)

# Load the model and scaler
model = load_model('fraud_detection_model1.keras')
scaler = joblib.load('scaler1.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    logging.info(f'Received data: {data}')
    
    if not data:
        logging.error('No data provided')
        return jsonify({'error': 'No data provided'}), 400

    # Extract required fields
    amount = data.get('amount')
    credit_score = data.get('credit_score')
    balance = data.get('balance')

    if amount is None or credit_score is None or balance is None:
        logging.error('Missing required fields')
        return jsonify({'error': 'Missing required fields'}), 400

    # Simulate additional features required by the model
    transaction_data = {
        'newbalanceOrig': balance - amount,
        'oldbalanceDest': np.random.uniform(0, 10000),
        'newbalanceDest': np.random.uniform(0, 10000) + amount,
        'transaction_frequency': np.random.randint(1, 50),
        'transaction_recency': np.random.randint(1, 365),
        'distance': np.random.uniform(0, 1000),
        'unusual_activity_flag': int(np.random.random() < 0.1),
        'num_unique_devices': np.random.randint(1, 10),
        'num_unique_locations': np.random.randint(1, 10),
        'blacklist_whitelist_status': np.random.choice([0, 1]),
        'transaction_amount_deviation': np.random.uniform(0, 500),
        'account_status': np.random.choice([0, 1])
    }

    # Prepare the input data
    features = [
        'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'transaction_frequency',
        'transaction_recency', 'distance', 'unusual_activity_flag', 'num_unique_devices',
        'num_unique_locations', 'blacklist_whitelist_status', 'transaction_amount_deviation',
        'account_status'
    ]
  
