from flask import Flask, render_template, jsonify, request
import numpy as np
from tensorflow import keras
import pickle

app = Flask(__name__)

car_price_prediction_model = pickle.load(open("mlmodels/car_price_prediction.pickle", "rb"))
stroke_prediction_model = pickle.load(open('mlmodels/stroke_prediction.pickle', 'rb'))
breast_cancer_prediction_model = keras.models.load_model('mlmodels/breast_cancer_prediction.h5')

@app.route('/')
def home():
    return render_template('index.html', title="CSC 120 Compilation")

@app.route('/linear-regression')
def lr():
    return render_template('linear-regression.html', title="Linear Regression")

@app.route('/logistic-regression')
def logistic_regression():
    return render_template('logisticregression.html', title="Logistic Regression")

@app.route('/neural-network')
def Vanilla_Neural_Network():
    return render_template('vanillaneuralnetwork.html', title="Vanilla Neural Network")

@app.route('/deep-learning')
def Deep_Learning():
    return render_template('deeplearning.html', title="Deep Learning")

@app.route('/reinforcement-learning')
def Deep_Reinforcement_learning():
    return render_template('deepreinforcementlearning.html', title="Deep Reinforcement Learning")

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
    data_x = np.array([data['engineSize'], data['boreratio'], data['compressionratio'], data['horsepower'],
                      data['peakrpm'], data['citympg'], data['highwaympg']]).reshape(-1, 7)
    prediction = car_price_prediction_model.predict(data_x)
    return jsonify({'prediction': prediction[0]})

@app.route('/predict/stroke-prediction', methods=['POST'])
def predict_stroke():
    data = request.get_json()
    data_x = np.array([data['age'], data['heart_disease'], data['work_type'], data['avg_glucose_level'],
                      data['bmi']]).reshape(-1, 5)
    prediction = stroke_prediction_model.predict_proba(data_x)[:, 1]
    return jsonify({'prediction': prediction[0]})

@app.route('/predict/breast-cancer-prediction', methods=['POST'])
def predict_breast_cancer():
    request_data = request.get_json()

    data = np.array([
        request_data['radius_mean'], request_data['texture_mean'], request_data['perimeter_mean'],
        request_data['area_mean'], request_data['smoothness_mean'], request_data['compactness_mean'],
        request_data['concavity_mean'], request_data['concave_points_mean'], request_data['symmetry_mean'],
        request_data['fractal_dimension_mean'], request_data['radius_se'], request_data['texture_se'],
        request_data['perimeter_se'], request_data['area_se'], request_data['smoothness_se'],
        request_data['compactness_se'], request_data['concavity_se'], request_data['concave_points_se'],
        request_data['symmetry_se'], request_data['fractal_dimension_se'], request_data['radius_worst'],
        request_data['texture_worst'], request_data['perimeter_worst'], request_data['area_worst'],
        request_data['smoothness_worst'], request_data['compactness_worst'], request_data['concavity_worst'],
        request_data['concave_points_worst'], request_data['symmetry_worst'], request_data['fractal_dimension_worst']
    ]).reshape(-1, 30)

    prediction = breast_cancer_prediction_model.predict(data)

    proba = float(prediction[0])
    diagnosis = "Benign" if proba < 0.5 else "Malignant"

    return jsonify({'probability': proba, 'diagnosis': diagnosis})

if __name__ == '__main__':
    app.run(debug=True)
