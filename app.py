from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load model at startup
model = joblib.load('fast_model.pkl')
print("✅ Model loaded!")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        symbol = data.get('symbol')
        features = data.get('features')
        
        if not features or len(features) != 13:
            return jsonify({'error': 'Need 13 features'}), 400
        
        X = np.array([features], dtype=np.float32)
        prediction = int(model.predict(X)[0])
        
        try:
            proba = model.predict_proba(X)[0]
            confidence = float(max(proba))
        except:
            confidence = 0.5
        
        return jsonify({
            'symbol': symbol,
            'prediction': prediction,
            'direction': 'UP' if prediction == 1 else 'DOWN',
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'model_type': 'RandomForestClassifier',
        'expected_features': 13,
        'feature_names': [
            'open', 'high', 'low', 'close', 'volume',
            'change (%)', 'last day change (%)',
            'return_1d', 'return_5d', 'return_10d',
            'volatility_10d', 'SMA_5', 'SMA_20'
        ]
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
```

4. **Click "Commit changes"**

---

#### **File 2: requirements.txt**

1. Click **"Add file"** → **"Create new file"**
2. **Name:** `requirements.txt`
3. **Paste:**
```
flask==3.0.0
flask-cors==4.0.0
scikit-learn==1.3.2
numpy==1.24.3
joblib==1.3.2
gunicorn==21.2.0
```

4. **Click "Commit changes"**

---

#### **File 3: Procfile**

1. Click **"Add file"** → **"Create new file"**
2. **Name:** `Procfile` (no extension!)
3. **Paste:**
```
web: gunicorn app:app
```

4. **Click "Commit changes"**

---

### **STEP 4: Upload Model to GitHub**

1. In your repo, click **"Add file"** → **"Upload files"**
2. **Drag & drop** `intraday_model.pkl` (that you downloaded from Kaggle)
3. **Click "Commit changes"**

Your repo should now have:
```
stock-prediction-backend/
├── app.py
├── requirements.txt
├── Procfile
├── intraday_model.pkl
└── README.md
