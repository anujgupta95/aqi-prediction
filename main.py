import pickle
import numpy as np
import json
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
print(MODEL_PATH)

with open(MODEL_PATH, 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    scaler = data['scaler']

@app.route('/')
def index():
    return "Welcome to the AQI Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = np.array([
            [float(data['co']), float(data['temperature']), float(data['humidity'])]
        ])
        
        # Scale and predict
        input_scaled = scaler.transform(input_data)
        input_reshaped = input_scaled.reshape(1, 1, input_scaled.shape[1])
        prediction = model.predict(input_reshaped)
        result = scaler.inverse_transform(prediction)
        
        response = {
            "CO_AQI": float(result[0][0]),
            "Temperature": float(result[0][1]),
            "Humidity": float(result[0][2])
        }
         
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500  

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 4000))  # Use Render's port or default locally
    app.run(host='0.0.0.0', port=port)