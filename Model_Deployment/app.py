
from flask import Flask, render_template, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np

app=Flask(__name__)
# Load model_prediction
mul_reg = open("Multiple_Regression_Student.pkl", "rb")
ml_model = joblib.load(mul_reg)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            absences = int(request.form['absences'])
            G1 = int(request.form['G1'])
            G2 = int(request.form['G2'])
            pred_args = [absences,G1,G2]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1, -1)
            model_prediction=ml_model.predict(pred_args_arr)
            model_prediction = round(int(model_prediction), 2)
        except valueError:
            raise "Enter the values correctly..."
    return render_template('predict.html', prediction = model_prediction)

if __name__ == '__main__':
    app.run()
