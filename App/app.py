from flask import Flask, flash, request, redirect, render_template, url_for
from werkzeug.utils import secure_filename

import os
import pickle
import pandas as pd
import numpy as np

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
# Load the configuration
# app.config['DEBUG'] = True

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
def params():
    return render_template('params.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    if 'choose_file' not in request.files:
        flash('No file part')
        # return to home page
        return redirect(url_for('home'))
    file = request.files['choose_file']

    # If user does not select file, browser submits an empty file without filename
    if file and allowed_file(file.filename):
        data = pd.read_csv(file)
        length = len(data)
        gen_model = pickle.load(open('genetic_model_def.pkl', 'rb'))
        sub_model = pickle.load(open('subclass_model_def.pkl', 'rb'))
        gen_prediction = gen_model.predict(data)
        sub_prediction = sub_model.predict(data)

        if length > 1:
            df = pd.DataFrame(data, columns=data.columns)
            data = data.reshape(1, -1)
            gen_prediction = gen_model.predict(data)
            sub_prediction = sub_model.predict(data)
            gen_result = df.inverse_transform(gen_prediction)
            sub_result = df.inverse_transform(sub_prediction)
            df.insert(0, 'Genetic_disorder', gen_result)
            df.insert(0, 'Disorder_subclass', sub_result)
            return render_template('predict_table.html', tables=[df.to_html(classes='data', header='True')])
        
        elif length == 1:
            return render_template('predict.html', disorder = data.inverse_transform(gen_prediction),  subclass = data.inverse_transform(sub_prediction))

if __name__ == '__main__':
    app.run(debug=True)
 

        
        