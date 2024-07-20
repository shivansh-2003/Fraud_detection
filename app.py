from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.exceptions import NotFittedError

app = Flask(__name__)

# Load the model and scaler
model = load_model('fraud_detection_model1.keras')
scaler = joblib.load('scaler1.pkl')

def preprocess_transaction(transaction, scaler):
    features = [
        'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
        'transaction_frequency', 'transaction_recency', 'distance',
        'unusual_activity_flag', 'num_unique_devices', 'num_unique_locations',
        'blacklist_whitelist_status', 'transaction_amount_deviation', 'account_status'
    ]

    # Ensure transaction dictionary has all required features
    assert all(feature in transaction for feature in features), "Missing features in the transaction data"

    transaction_values = np.array([transaction[feature] for feature in features]).reshape(1, -1)
    try:
        transaction_scaled = scaler.transform(transaction_values)
    except NotFittedError as e:
        print(f"Error: {e}")
        return None
    return transaction_scaled

def generate_transaction_score(transaction_scaled):
    transaction_scaled = np.array(transaction_scaled).reshape(1, -1)  # Ensure it’s a 2D array
    transaction_score = model.predict(transaction_scaled)[0][0]
    return transaction_score

def detect_fraud(transaction_score, threshold=0.3):
    return transaction_score >= threshold

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_input = data.get('user_input', {})

    enriched_transaction = {
        'amount': float(user_input.get('amount', 0)),
        'balance': float(user_input.get('balance', 0)),
        'credit_score': float(user_input.get('credit_score', 0)),
        'oldbalanceOrg': float(user_input.get('balance', 0)),
        'newbalanceOrig': float(user_input.get('balance', 0)) - float(user_input.get('amount', 0)),
        'oldbalanceDest': np.random.uniform(0, 10000),
        'newbalanceDest': np.random.uniform(0, 10000) + float(user_input.get('amount', 0)),
        'transaction_frequency': np.random.randint(1, 50),
        'transaction_recency': np.random.randint(1, 365),
        'distance': np.random.uniform(0, 1000),
        'unusual_activity_flag': 1 if np.random.random() < 0.1 else 0,
        'num_unique_devices': np.random.randint(1, 10),
        'num_unique_locations': np.random.randint(1, 10),
        'blacklist_whitelist_status': np.random.choice([0, 1]),
        'transaction_amount_deviation': np.random.uniform(0, 500),
        'account_status': np.random.choice([0, 1])
    }

    preprocessed_transaction = preprocess_transaction(enriched_transaction, scaler)

    if preprocessed_transaction is not None:
        transaction_score = generate_transaction_score(preprocessed_transaction)
        is_fraud = detect_fraud(transaction_score)
        return jsonify({
            'transaction_score': transaction_score,
            'fraud_detected': is_fraud
        })
    else:
        return jsonify({'error': 'Preprocessing error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', localhost=5000, debug=True)
