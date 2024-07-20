from flask import Flask, request, jsonify
from flask_talisman import Talisman

app = Flask(__name__)
Talisman(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    # Extract required fields
    amount = data.get('amount')
    credit_score = data.get('credit_score')
    balance = data.get('balance')

    if amount is None or credit_score is None or balance is None:
        return jsonify({'error': 'Missing required fields'}), 400

    # Mock prediction logic for testing
    transaction_score = (amount + credit_score + balance) / 3
    is_fraud = transaction_score > 1000

    return jsonify({
        'transaction_score': transaction_score,
        'is_fraud': is_fraud
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, ssl_context='adhoc')
