from flask import Flask, flash, request, redirect, render_template, url_for
from werkzeug.utils import secure_filename

import os
import pickle
import pandas as pd
import numpy as np

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
# Load the configuration
app.config['DEBUG'] = True

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'txt', 'xls', 'xlsx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route for the GitHub page
#@app.route('/<convertidor: nombre_variable>', methods=['POST'])
#def git_update():
#    repo = git.Repo('./Genomes_project')
#    origin = repo.remotes.origin
#    repo.create_head('main', origin.refs.main).set_tracking_branch(origin.refs.main).checkout()
#    origin.pull()
#    return "", 200

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/params', methods=['GET'])
def home():
    return render_template('params.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if 'choose_file' not in request.files:
        flash('No file part')
        return redirect(url_for(home))
    file = request.files['choose_file']

    # If user does not select file, browser submits an empty file without filename
    if file and allowed_file(file.filename):
        data = pd.read_csv(file)
        length = len(data)
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        scaled_data = scaler.transform(data)
        predicion = model.predict(scaled_data)

        if length > 1:
            df = pd.DataFrame(data, columns=data.columns)
            data = scaler.transform(data)
            data = data.reshape(1, -1)
            prediction = model.predict(data)
            prediction2 = model2.predict(data)
            result = df.inverse_transform(prediction)
            result2 = df.inverse_transform(prediction2)
            df.insert(0, 'prediction', result)
            df.insert(0, 'prediction2', prediction2)
            return render_template('predict_table.html', tables=[df.to_html(classes='data', header='True')])
        
        elif length == 1:
            return render_template('predict.html', result = predicion[0])

        
        