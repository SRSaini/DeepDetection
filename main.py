import os
from functions import pred
from flask import Flask, request, render_template, jsonify
from keras import backend as K

app = Flask(__name__)


@app.route('/')
def home():
    
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    

    if request.method == 'POST':
        
        if 'file' not in request.files:
            return 'No file found'
    user_file = request.files['file']

    if user_file.filename == '':
        return 'file name not found …'
    else:
        path = os.path.join(os.getcwd()+user_file.filename)
        user_file.save(path)
        K.clear_session()
        classes = pred(path)
        K.clear_session()

        return render_template(
            'submit.html',
            name="success",
            email=classes[0],
            site=str(classes[1]),
            comments=user_file)


