from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow frontend to access backend

# Load trained model
with open("fault_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = np.array([[data["voltage"], data["current"], data["temperature"], data["humidity"]]])
        
        prediction = model.predict(features)
        fault_status = "Fault Detected" if prediction[0] == 1 else "No Fault"

        response = {
            "voltage": data["voltage"],
            "current": data["current"],
            "temperature": data["temperature"],
            "humidity": data["humidity"],
            "fault": fault_status
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)