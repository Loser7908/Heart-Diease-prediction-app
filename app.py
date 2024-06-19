from flask import Flask, request, jsonify

import numpy as np
import pickle
import pandas as pd

# Load the model
model = pickle.load(open('heart_disease_prediction_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting and validating input values from the request
        required_fields = ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth',
                           'DiffWalking', 'Sex', 'AgeCategory', 'Diabetic', 'PhysicalActivity',
                           'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer']

        input_data = {}
        for field in required_fields:
            value = request.form.get(field)
            if value is None:
                return jsonify({'error': f'Missing value for {field}'})
            input_data[field] = value

        # Convert inputs to the appropriate types
        input_data = {
            'BMI': float(input_data['BMI']),
            'Smoking': int(input_data['Smoking']),
            'AlcoholDrinking': int(input_data['AlcoholDrinking']),
            'Stroke': int(input_data['Stroke']),
            'PhysicalHealth': float(input_data['PhysicalHealth']),
            'MentalHealth': float(input_data['MentalHealth']),
            'DiffWalking': int(input_data['DiffWalking']),
            'Sex': int(input_data['Sex']),
            'AgeCategory': int(input_data['AgeCategory']),
            'Diabetic': int(input_data['Diabetic']),
            'PhysicalActivity': int(input_data['PhysicalActivity']),
            'GenHealth': int(input_data['GenHealth']),
            'SleepTime': float(input_data['SleepTime']),
            'Asthma': int(input_data['Asthma']),
            'KidneyDisease': int(input_data['KidneyDisease']),
            'SkinCancer': int(input_data['SkinCancer']),
        }

        # Creating the input DataFrame
        input_df = pd.DataFrame([input_data])

        # Making prediction
        predicted_prob = model.predict_proba(input_df)[:, 1]
        predicted_class = model.predict(input_df)

        # Convert to percentage
        predicted_percentage = predicted_prob[0] * 100

        # Returning the result
        return jsonify({
            'HeartDiseasePrediction': int(predicted_class[0]),  # 0 or 1
            'HeartDiseaseProbability': f'{predicted_percentage:.2f}%'
        })
    except Exception as e:
        # Handling errors
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
