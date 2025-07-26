from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the scaler (if used for feature normalization)
try:
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    scaler = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract input values
        features = [float(request.form[key]) for key in request.form.keys()]
        input_data = pd.DataFrame([features], columns=[
            "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population",
            "AveOccup", "Latitude", "Longitude"
        ])

        # Apply scaling if scaler exists
        if scaler:
            input_data = scaler.transform(input_data)

        # Predict the house price
        prediction = model.predict(input_data)[0]

        # Adjust prediction range
        adjusted_prediction = prediction * 100000  # Scaling price appropriately

        return render_template("index.html", prediction_text=f"Estimated House Price: {prediction * 1000:.2f} RS")


    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
