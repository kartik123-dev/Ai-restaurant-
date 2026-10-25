
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Sample sales data for the Class 12 AI project
customers = np.array([45, 50, 48, 60, 75, 95, 85]).reshape(-1, 1)
sales = np.array([4500, 5000, 4800, 6000, 7500, 9500, 8500])

# Train the AI model
model = LinearRegression()
model.fit(customers, sales)


@app.route("/")
def home():
    return jsonify({
        "project": "Restaurant AI",
        "status": "Backend is running",
        "message": "Sales prediction API is ready"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    try:
        customer_count = float(data.get("customers", 0))

        if not 0 <= customer_count <= 100000:
            return jsonify({"error": "Customers must be between 0 and 100000"}), 400

        prediction = model.predict([[customer_count]])[0]

        return jsonify({
            "customers": int(customer_count),
            "predicted_sales": round(max(0, float(prediction)), 2),
            "currency": "INR"
        })

    except (TypeError, ValueError):
        return jsonify({"error": "Please enter a valid customer count"}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
  
