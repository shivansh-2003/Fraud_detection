# Fraud Detection System

A machine learning-based fraud detection API that predicts whether financial transactions are fraudulent using a deep neural network model trained on transaction data.

## Overview

This system provides a REST API endpoint for real-time fraud detection on financial transactions. It uses a trained Keras neural network model that analyzes transaction patterns and account characteristics to generate fraud risk scores.

**Key Features:**
- Real-time fraud prediction via REST API
- Deep neural network with batch normalization and dropout regularization
- SMOTE-based handling of class imbalance
- Comprehensive feature engineering pipeline
- Easy deployment with Flask and CORS support

## Project Structure

```
.
├── app.py              # Flask API server
├── mete.ipynb          # Model training and evaluation notebook
├── requirements.txt    # Python dependencies
├── fraud_detection_model1.keras  # Trained model (binary)
├── scaler1.pkl         # Fitted StandardScaler (binary)
└── README.md           # This file
```

## Installation

### Prerequisites
- Python 3.7+
- pip

### Setup

1. Clone or download the project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the model and scaler files are in the project directory:
   - `fraud_detection_model1.keras`
   - `scaler1.pkl`

## Usage

### Starting the API Server

```bash
python app.py
```

The server will start on `http://0.0.0.0:5000`

### API Endpoint

**POST** `/predict`

#### Request

```json
{
  "amount": 150.50,
  "credit_score": 750,
  "balance": 5000.00
}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `amount` | float | Yes | Transaction amount |
| `credit_score` | float | Yes | Customer's credit score (typically 300-850) |
| `balance` | float | No | Current account balance |

#### Response

```json
{
  "transaction_score": 0.25,
  "is_fraud": false
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `transaction_score` | float | Fraud probability score (0.0 - 1.0) |
| `is_fraud` | boolean | Fraud prediction (threshold: 0.3) |

#### Example Usage

**cURL:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 500, "credit_score": 680, "balance": 2500}'
```

**Python:**
```python
import requests

url = "http://localhost:5000/predict"
data = {
    "amount": 500,
    "credit_score": 680,
    "balance": 2500
}

response = requests.post(url, json=data)
print(response.json())
```

**JavaScript:**
```javascript
fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    amount: 500,
    credit_score: 680,
    balance: 2500
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

## Model Architecture

The fraud detection model is a deep neural network with the following architecture:

- **Input Layer:** 12 features
- **Hidden Layers:**
  - Dense(128) → BatchNormalization → Dropout(0.3)
  - Dense(64) → BatchNormalization → Dropout(0.3)
  - Dense(32) → BatchNormalization → Dropout(0.3)
  - Dense(16) → BatchNormalization
- **Output Layer:** Dense(1, activation='sigmoid')

**Compilation:**
- Optimizer: Adam (learning rate: 0.001)
- Loss: Binary Crossentropy
- Metrics: Accuracy

**Total Parameters:** 13,505 (52.75 KB)

## Features Used

The model uses 12 engineered features for prediction:

1. `transaction_amount` - The monetary value of the transaction
2. `transaction_frequency` - Number of transactions in recent period
3. `distance` - Geographical distance of transaction
4. `account_age_days` - Days since account creation
5. `transaction_recency` - Days since last transaction
6. `unusual_activity_flag` - Binary flag for unusual patterns
7. `num_unique_devices` - Number of unique devices used
8. `num_unique_locations` - Number of unique transaction locations
9. `blacklist_whitelist_status` - Categorical status (0: blacklist, 1: whitelist)
10. `transaction_amount_deviation` - Deviation from average transaction amount
11. `credit_score` - Customer's credit score
12. `account_status` - Categorical status (0: inactive, 1: active)

## Training Pipeline

### Data Processing

The training pipeline (`mete.ipynb`) includes:

1. **Data Loading:** Reads fraud transaction dataset
2. **Feature Selection:** Extracts 12 essential features
3. **Encoding:** LabelEncoding for categorical features
4. **Class Balancing:** SMOTE for handling class imbalance (original: 50.9% fraud)
5. **Scaling:** StandardScaler normalization
6. **Train-Test Split:** 80-20 split

### Model Training

- **Cross-Validation:** 5-fold Stratified K-Fold
- **Epochs:** Up to 50 with early stopping (patience: 10)
- **Batch Size:** 32
- **Class Weights:** Computed for balanced learning
- **Learning Rate:** Adaptive with ReduceLROnPlateau

## Files Description

### app.py
Flask REST API server with:
- CORS support for cross-origin requests
- Model and scaler loading
- `/predict` endpoint for fraud predictions
- Feature simulation and scaling pipeline

### mete.ipynb
Jupyter notebook containing:
- Complete data preprocessing pipeline
- SMOTE implementation for class balancing
- Model architecture definition
- Training with stratified k-fold cross-validation
- Model and scaler serialization

### requirements.txt
Python package dependencies:
- Flask & flask-cors: REST API framework
- TensorFlow: Deep learning framework
- scikit-learn: ML utilities and preprocessing
- joblib: Model serialization
- numpy & pandas: Data processing
- gunicorn: Production WSGI server

## Configuration

### Fraud Threshold
The default fraud detection threshold is **0.3**. Adjust in `app.py`:
```python
is_fraud = transaction_score > 0.3  # Modify this value
```

- **Lower threshold (e.g., 0.2):** More sensitive, more false positives
- **Higher threshold (e.g., 0.5):** Less sensitive, more false negatives

## Deployment

### Local Development
```bash
python app.py
```

### Production Deployment (Gunicorn)
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## Performance Notes

- Model size: ~52 KB
- Prediction latency: ~8-11ms per transaction
- Throughput: ~100 transactions/second per core (with gunicorn)

## Error Handling

### Missing Required Fields
```json
{
  "error": "Missing required fields"
}
```

### No Data Provided
```json
{
  "error": "No data provided"
}
```

## Limitations

- Features like `account_age_days`, `transaction_frequency`, etc. are simulated in the API. For production, integrate with actual transaction history data
- The model is trained on synthetic data and should be validated on real-world transactions
- Threshold of 0.3 may need tuning based on business requirements

## Future Improvements

- Real transaction history integration
- Explainability features (SHAP values)
- Model retraining pipeline
- Batch prediction endpoint
- Monitoring and logging
- Redis caching for repeated predictions
- WebSocket support for streaming predictions

## Dependencies

See `requirements.txt` for complete list:
- Flask >= 2.0
- TensorFlow >= 2.10
- scikit-learn >= 1.0
- numpy >= 1.20
- pandas >= 1.3
- joblib >= 1.0

## License

[Specify your license here]

## Author

[Your name/organization]

## Support

For issues or questions, please contact [support contact information]
