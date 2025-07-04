import json
import pickle
import os
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'remodel.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaling.pkl')

# Load model and scaler
remodel = pickle.load(open(MODEL_PATH, 'rb'))
scalar = pickle.load(open(SCALER_PATH, 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

# POST API for Postman JSON requests
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()['data']  # Use .get_json() safely
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = remodel.predict(new_data)
    return jsonify({'prediction': output[0]})

# HTML form handler
@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    output = remodel.predict(final_input)[0]
    return render_template("home.html", prediction_text=f"The House price prediction is {output:.2f}")

if __name__ == "__main__":
    app.run(debug=True)
