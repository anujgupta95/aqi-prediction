import pickle
import numpy as np
from http.server import BaseHTTPRequestHandler
import json
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model.pkl')

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = json.loads(self.rfile.read(content_length))
            
            # Load model
            with open(MODEL_PATH, 'rb') as f:
                model, scaler = pickle.load(f)
            
            # Process input
            input_data = np.array([
                [float(post_data['co']), 
                 float(post_data['temperature']), 
                 float(post_data['humidity'])]
            ])
            
            # Predict
            input_scaled = scaler.transform(input_data)
            input_reshaped = input_scaled.reshape(1, 1, input_scaled.shape[1])
            prediction = model.predict(input_reshaped)
            result = scaler.inverse_transform(prediction)
            
            # Return response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "CO_AQI": float(result[0][0]),
                "Temperature": float(result[0][1]),
                "Humidity": float(result[0][2])
            }).encode())
            
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

# For local testing
if __name__ == '__main__':
    from http.server import HTTPServer
    server = HTTPServer(('localhost', 8000), handler)
    server.serve_forever()
