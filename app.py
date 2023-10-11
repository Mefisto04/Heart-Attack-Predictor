import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, render_template,jsonify

app = Flask(__name__)

# Load your trained model and data
heart_data = pd.read_csv('heart_disease_data.csv')
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
model = LogisticRegression()
model.fit(X, Y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/map.html')
def map():
    return render_template('map.html')

@app.route('/avatar.html')
def avatar():
    return render_template('avatar.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect and preprocess input values
    age = float(request.form['age'])
    sex = float(request.form['sex'])
    cp = float(request.form['cp'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    fbs = float(request.form['fbs'])
    restecg = float(request.form['restecg'])
    thalach = float(request.form['thalach'])
    exang = float(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = float(request.form['slope'])
    ca = float(request.form['ca'])
    thal = float(request.form['thal'])

    # Make a prediction using the model
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)

    # result = "You are likely to have a heart attack" if prediction[
    #                                                         0] == 1 else "You are not likely to have a heart attack"
    #
    # return f"Prediction: {result}"
    result = "You have a heart attack" if prediction[0] == 1 else "You do not have a heart attack"

    # Return the prediction as plain text
    return result


if __name__ == '__main__':
    app.run(debug=True)
