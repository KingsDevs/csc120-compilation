from flask import Flask, render_template, jsonify, request
import numpy as np
from tensorflow import keras
from PIL import Image
import pickle
import base64
from io import BytesIO
import random
import sqlite3

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html', title="CSC 120 COMPILATION")

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

@app.route('/concrete-strength-prediction-nn')
def concrete_strength_prediction_nn():
    # if request.method == 'POST':
    #     data = request.form

    #     cement = float(data.get('cement'))
    #     slag = float(data.get('slag'))
    #     flyash = float(data.get('flyash'))
    #     water = float(data.get('water'))
    #     superplasticizer = float(data.get('superplasticizer'))
    #     coarseaggregate = float(data.get('coarseaggregate'))
    #     fineaggregate = float(data.get('fineaggregate'))
    #     age = float(data.get('age'))

    #     data_x = np.array([cement, slag, flyash, water, superplasticizer, coarseaggregate, fineaggregate, age]).reshape(-1, 8)
        
    #     concrete_strength_prediction_nnmodel = keras.models.load_model("mlmodels/concrete_strength_prediction_nn.h5")

    #     prediction = concrete_strength_prediction_nnmodel.predict(data_x)[0]
    #     prediction = format(prediction[0], '.3f')

    #     return render_template("concretestrengthprediction_nn_result.html", title="Concrete Strength Prediction Deep Learning Ver", prediction=prediction)


    return render_template("concretestrengthprediction_nn.html", title="Concrete Strength Prediction Deep Learning Ver")

@app.route('/concrete-strength-prediction-lr')
def concrete_strength_prediction_lr():
    return render_template("concretestrengthprediction_lr.html", title="Concrete Strength Prediction Linear Regression Ver")

@app.route('/stroke-prediction')
def stroke_prediction():
    return render_template("strokeprediction.html", title="Stroke Probability Prediction")

@app.route('/diabetes-prediction-lr')
def diabetes_prediction_lr():
    return render_template("diabetesprediction-lr.html", title="Diabetes Prediction Logistical Regression Ver")

@app.route('/diabetes-prediction-nn')
def diabetes_prediction_nn():
    # if request.method == "POST":
    #     data = request.form
    #     data_x = np.array([int(data.get('gender')), int(data.get('age')), int(data.get('hypertension')), int(data.get('heart-disease')), int(data.get('smoke-history')), float(data.get('bmi')),
    #                     float(data.get('HbA1c-level')), float(data.get('blood-glucose-level'))]).reshape(-1, 8)
        
    #     diabetes_prediction_model_nn = keras.models.load_model("mlmodels/diabetes_prediction_nn.h5")
    #     prediction = diabetes_prediction_model_nn.predict(data_x)[0]

    #     return render_template("diabetesprediction-nn-result.html", title="Diabetes Prediction Neural Network Regression Ver", prediction=prediction[0])

    return render_template("diabetesprediction-nn.html", title="Diabetes Prediction Neural Network Ver")

@app.route('/breast-cancer-prediction')
def breast_cancer_prediction():
    return render_template('breastcancerprediction.html', title="Breast Cancer Prediction")

@app.route('/tomato-leaf-disease-prediction')
def tomato_leaf_disease_prediction():
    return render_template("tomatoleafdiseaseprediction.html", title="Tomato Leaf Disease Prediction")

@app.route('/rl')
def serve_webgl():
    return render_template("rl-game.html", title="Autonomous Drone Navigation")


@app.route('/predict/car-price-prediction', methods=['POST'])
def predict_car_price():
    data = request.get_json()
    data_x = np.array([data['engineSize'], data['boreratio'], data['compressionratio'], data['horsepower'],
                      data['peakrpm'], data['citympg'], data['highwaympg']]).reshape(-1, 7)
    
    car_price_prediction_model = pickle.load(open("mlmodels/car_price_prediction.pickle", "rb"))
    prediction = car_price_prediction_model.predict(data_x)
    return jsonify({'prediction': prediction[0]})

@app.route('/predict/concrete-strength-prediction-lr', methods=['POST'])
def predict_concrete_strength_lr():
    data = request.get_json()
    data_x = np.array([data['cement'], data['slag'], data['flyash'], data['water'], data['superplasticizer'],
                      data['coarseaggregate'], data['fineaggregate'], data['age']]).reshape(-1, 8)
    
    concrete_strength_prediction_lrmodel = pickle.load(open("mlmodels/concrete_strength_prediction_lr.pickle", "rb"))
    prediction = concrete_strength_prediction_lrmodel.predict(data_x)
    return jsonify({'csMpa': prediction[0]})

@app.route('/predict/concrete-strength-prediction-nn', methods=['POST'])
def predict_concrete_strength_nn():
    data = request.get_json()
    data_x = np.array([data['cement'], data['slag'], data['flyash'], data['water'], data['superplasticizer'],
                      data['coarseaggregate'], data['fineaggregate'], data['age']]).reshape(-1, 8)
    
    concrete_strength_prediction_nnmodel = keras.models.load_model("mlmodels/concrete_strength_prediction_nn.h5")
    prediction = concrete_strength_prediction_nnmodel.predict(data_x)[0]
    prediction = format(prediction[0], '.3f')

    return jsonify({'csMpa': prediction})

@app.route('/predict/stroke-prediction', methods=['POST'])
def predict_stroke():
    data = request.get_json()
    data_x = np.array([data['age'], data['heart_disease'], data['work_type'], data['avg_glucose_level'],
                      data['bmi']]).reshape(-1, 5)
    
    stroke_prediction_model = pickle.load(open('mlmodels/stroke_prediction.pickle', 'rb'))
    prediction = stroke_prediction_model.predict_proba(data_x)[:, 1]
    return jsonify({'prediction': prediction[0]})

@app.route('/predict/diabetes-prediction-lr', methods=['POST'])
def predict_diabetes_lr():
    data = request.get_json()
    data_x = np.array([data['gender'], data['age'], data['hypertension'], data['heart_disease'], data['smoke_history'], data['bmi'],
                      data['HbA1c_level'], data['blood_glucose_level']]).reshape(-1, 8)
    
    # from sklearn import preprocessing
    # stand = preprocessing.StandardScaler()
    # data_x = stand.fit_transform(data_x)

    diabetes_prediction_model_lr = pickle.load(open('mlmodels/diabetes_prediction_logr.pickle', 'rb'))

    prediction = diabetes_prediction_model_lr.predict_proba(data_x)[:, 1]
    return jsonify({'prediction': prediction[0]})

@app.route('/predict/diabetes-prediction-nn', methods=['POST'])
def predict_diabetes_nn():
    data = request.get_json()
    data_x = np.array([data['gender'], data['age'], data['hypertension'], data['heart_disease'], data['smoke_history'], data['bmi'],
                      data['HbA1c_level'], data['blood_glucose_level']]).reshape(-1, 8)

    diabetes_prediction_model_nn = keras.models.load_model("mlmodels/diabetes_prediction_nn.h5")
    prediction = diabetes_prediction_model_nn.predict(data_x)

    return jsonify({'prediction': float(prediction[0])})

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

    breast_cancer_prediction_model = keras.models.load_model('mlmodels/breast_cancer_prediction.h5')
    prediction = breast_cancer_prediction_model.predict(data)

    proba = float(prediction[0])
    diagnosis = "Benign" if proba < 0.5 else "Malignant"

    return jsonify({'probability': proba, 'diagnosis': diagnosis})

@app.route('/predict/tomato-leaf-disease-prediction', methods=['POST'])
def predict_tomato_leaf_disease():
    imageFetched = request.form.get('isImageFetched')

    image = None
    if imageFetched == 'true':
        image_file = request.form.get('image')
        image_data = base64.b64decode(image_file)
        image = Image.open(BytesIO(image_data))
    else:
        image_file = request.files['image']
        image = Image.open(image_file)

    model_index = int(request.form.get('model'))
    if model_index < 0: 
        model_index = 0

    if model_index == 0:
        model = keras.models.load_model('mlmodels/tomato_leaf_disease_detection.h5')
    elif model_index == 1:
        model = keras.models.load_model('mlmodels/tomato_leaf_disease_detection_inception.h5')
    else:
        model = keras.models.load_model('mlmodels/pretrain_model.h5')
    
    image = image.resize((112, 112))
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)

    return jsonify({
        'Bacterial Spot': float(predictions[0][0]),
        'Early Blight': float(predictions[0][1]),
        'Late Blight': float(predictions[0][2]),
        'Leaf Mold': float(predictions[0][3]),
        'Septoria Leaf Spot': float(predictions[0][4]),
        'Two-spotted Spider Mite': float(predictions[0][5]),
        'Target Spot': float(predictions[0][6]),
        'Yellow Leaf Curl Virus': float(predictions[0][7]),
        'Mosaic Virus': float(predictions[0][8]),
        'Healthy': float(predictions[0][9])
    })

@app.route('/fetch-image-tomato-leaf-disease', methods=['POST'])
def fetch_image_tomato_leaf_disease():
    # label = random.choice(["bacteria_spot", "early_blight", "healthy", "late_blight", "leaf_mold", "mosaic_virus", "septoria_leaf_spot", "spider_mites_two_spotted_mite", "target_spot", "yellow_leaf_curl_virus"])

    # # Get the absolute path of the directory containing your script
    # script_dir = os.path.dirname(os.path.abspath(__file__))

    # # Construct the absolute path to the image
    # image_path = os.path.join(script_dir, "static", "images", "tomato_leaf_images", label, f"{label}_{random.choice([1,2,3,4,5,6,7,8,9,10])}.jpg")

    conn = sqlite3.connect('tomato_leaf_images.db')
    cursor = conn.cursor()

    # Execute the select query to get a random image
    select_query = "SELECT * FROM images ORDER BY RANDOM() LIMIT 1"
    cursor.execute(select_query)

    result = cursor.fetchone()

    # Close the connection
    conn.close()

    id, image_file, label = result

    # image_file = open(image_path, 'rb')
    encoded_image = base64.b64encode(image_file).decode('utf-8')
    # image_file.close()
    

    return jsonify({
        'image': encoded_image,
        'label': label
    })

@app.route('/about')
def about():
    return render_template("about.html", title="About")


if __name__ == '__main__':
    app.run(debug=True)


