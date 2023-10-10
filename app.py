import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import streamlit as st
from PIL import Image


# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('heart_disease_data.csv')
heart_data.head()
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
model = LogisticRegression()
# training the LogisticRegression model with Training data
model.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

def predict_heart_attack(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })

    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit web app
st.title('Heart Attack Prediction')
st.sidebar.header('Input Features')

# Input features
age = st.sidebar.slider('Age', 29, 77, 40)
sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
cp = st.sidebar.slider('Chest Pain Type (cp)', 0, 3, 1)
trestbps = st.sidebar.slider('Resting Blood Pressure (trestbps)', 94, 200, 120)
chol = st.sidebar.slider('Cholesterol (chol)', 126, 564, 200)
fbs = st.sidebar.selectbox('Fasting Blood Sugar (fbs)', ['< 120 mg/dl', '>= 120 mg/dl'])
restecg = st.sidebar.slider('Resting Electrocardiographic Results (restecg)', 0, 2, 1)
thalach = st.sidebar.slider('Maximum Heart Rate Achieved (thalach)', 71, 202, 150)
exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', ['No', 'Yes'])
oldpeak = st.sidebar.slider('ST Depression Induced by Exercise Relative to Rest (oldpeak)', 0.0, 6.2, 1.0)
slope = st.sidebar.slider('Slope of the Peak Exercise ST Segment (slope)', 0, 2, 1)
ca = st.sidebar.slider('Number of Major Vessels (0-3) Colored by Flourosopy (ca)', 0, 3, 1)
thal = st.sidebar.slider('Thalassemia (thal)', 0, 2, 1)

# Convert categorical inputs to numerical values
sex = 1 if sex == 'Male' else 0
fbs = 1 if fbs == '>= 120 mg/dl' else 0
exang = 1 if exang == 'Yes' else 0

# Prediction
if st.button('Predict'):
    result = predict_heart_attack(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    if result == 1:
        st.error('The person is predicted to have a heart attack.')
    else:
        st.success('The person is predicted to be healthy.')

st.sidebar.text('Developed by Mefisto')