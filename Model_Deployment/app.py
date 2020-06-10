
from flask import Flask, render_template, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__)
# Load model_prediction
ml_model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
            absences = int(request.form['absences'])
            G1 = int(request.form['G1'])
            G2 = int(request.form['G2'])
            pred_args = [absences,G1,G2]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1, -1)
            model_prediction=ml_model.predict(pred_args_arr)
            model_prediction = round(int(model_prediction), 2)

    return render_template('result.html', prediction = model_prediction)

if __name__ == '__main__':
    app.run(debug=True)
