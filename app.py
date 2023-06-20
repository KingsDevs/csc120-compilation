from flask import Flask, render_template, jsonify, request
import numpy as np
import pickle


app = Flask(__name__)

car_price_prediction_model = None


@app.route('/')
def home():
    return render_template('index.html', title="CSC 120 Compilation")

@app.route('/linear-regression')
def lr():
    return render_template('linear-regression.html', title="Linear Regression")

@app.route('/car-price-prediction')
def car_price_prediction():
    return render_template('carpriceprediction.html', title="Car Price Prediction")

@app.route('/rl')
def serve_webgl():
    return render_template("rl-game.html")


@app.route('/predict/car-price-prediction')
def predict_car_price():
    data = request.get_json()
    data = [data['engineSize'], data['boreratio'], data['compressionratio'], data['horsepower'], data['peakrpm'], data['citympg'], data['highwaympg']]
    data_x = np.array(data).reshape(-1, 7)

    prediction = car_price_prediction_model.predict(data_x)

    return jsonify({'prediction': prediction[0]})

def load_models():
    model_file = open("mlmodels/car_price_prediction.pickle", "rb")
    car_price_prediction_model = pickle.load(model_file)


if __name__ == '__main__':
    load_models()
    app.run(debug=True)

