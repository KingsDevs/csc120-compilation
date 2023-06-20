from flask import Flask, render_template, jsonify, request
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from tensorflow import keras
import pickle


app = Flask(__name__)

model_file = open("mlmodels/car_price_prediction.pickle", "rb")
car_price_prediction_model = pickle.load(model_file)

model_file = open('mlmodels/stroke_prediction.pickle', 'rb')
stroke_prediction_model = pickle.load(model_file)

breast_cancer_prediction_model = keras.models.load_model('mlmodels/breast_cancer_prediction.h5')


@app.route('/')
def home():
    return render_template('index.html', title="CSC 120 Compilation")

@app.route('/linear-regression')
def lr():
    return render_template('linear-regression.html', title="Linear Regression")

@app.route('/car-price-prediction')
def car_price_prediction():
    return render_template('carpriceprediction.html', title="Car Price Prediction")

@app.route('/stroke-prediction')
def stroke_prediction():
    return render_template("strokeprediction.html", title="Stroke Probability Prediction")

@app.route('/breast-cancer-prediction')
def breast_cancer_prediction():
    return render_template('breastcancerprediction.html', title="Breast Cancer Prediction")

@app.route('/tomato-leaf-disease-prediction')
def tomato_leaf_disease_prediction():
    return render_template("tomatoleafdiseaseprediction.html", title="Tomato Leaf Disease Prediction")

@app.route('/rl')
def serve_webgl():
    return render_template("rl-game.html")


@app.route('/predict/car-price-prediction', methods=['POST'])
def predict_car_price():
    data = request.get_json()
    data = [data['engineSize'], data['boreratio'], data['compressionratio'], data['horsepower'], data['peakrpm'], data['citympg'], data['highwaympg']]
    data_x = np.array(data).reshape(-1, 7)

    prediction = car_price_prediction_model.predict(data_x)

    return jsonify({'prediction': prediction[0]})

@app.route('/predict/stroke-prediction', methods=['POST'])
def predict_stroke():
    data = request.get_json()
    data = [data['age'], data['heart_disease'], data['work_type'], data['avg_glucose_level'], data['bmi']]

    data_x = np.array(data).reshape(-1, 5)
    prediction = stroke_prediction_model.predict_proba(data_x)[:,1]

    return jsonify({'prediction': prediction[0]})

@app.route('/predict/breast-cancer-prediction', methods=['POST'])
def predict_breast_cancer():
    col_names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']
    request_data = request.get_json()

    data = []
    for col in col_names:
        data.append(request_data[col])
    data = np.array(data).reshape(-1, 30)

    prediction = breast_cancer_prediction_model.predict(data)

    proba = float(prediction[0])
    diagnosis = "Benign"

    if proba >= 0.5: diagnosis = "Malignant"

    return jsonify({'probability': proba, 'diagnosis': diagnosis})



if __name__ == '__main__':
    # model_file = open("mlmodels/car_price_prediction.pickle", "rb")
    # car_price_prediction_model = pickle.load(model_file)
    app.run(debug=True)

