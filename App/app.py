from flask import Flask, flash, request, redirect, render_template, url_for
from werkzeug.utils import secure_filename

import os
import pickle
import pandas as pd
import numpy as np
import joblib

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)

# Load the model
encoder = joblib.load(open('encoder.pkl', 'rb'))
encoder_gen = joblib.load(open('encoder_gen.pkl', 'rb'))
gen_model = pickle.load(open('genetic_model_def.pkl', 'rb'))
sub_model = pickle.load(open('subclass_model_def.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/params', methods=['GET'])
def params():
    return render_template('params.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    print(int_features)
    final_features = np.array(int_features).reshape(1, -1)
    gen_prediction = gen_model.predict(final_features)
    sub_prediction = sub_model.predict(final_features)
    pred1 = encoder_gen.inverse_transform([int(gen_prediction)])
    pred2 = encoder.inverse_transform([int(sub_prediction)])
    return render_template('index.html', prediction_text = f'Patient suffers from the disease group: {pred1[0]}. The specific condition is named "{pred2[0]}".')

if __name__ == '__main__':
    app.run(port=80)
 

        
        