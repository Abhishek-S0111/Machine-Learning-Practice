from flask import Flask, request, jsonify
from catboost import CatBoostRegressor
from flask_cors import CORS
import numpy as np
import pandas as pd

def feature_engineering(df):
    #print(df)

    # Intensity index
    df['Intensity_Index'] = df['Heart_Rate'] / df['Duration']

    # Log transformations 
    df['Age'] = np.log1p(df['Age'])
    df['Body_Temp'] = np.log1p(df['Body_Temp'])

    # Basal Metabolic Rate
    df['BMR'] = (
        10 * df['Weight'] + 
        6.25 * df['Height'] - 
        5 * df['Age'] + 
        np.where(df['Sex'] == 1, 5, -161)
    )

    # Core interactions
    df['HR_Temp_Interaction'] = df['Heart_Rate'] * df['Body_Temp']
    df['HR_Duration_Interaction'] = df['Heart_Rate'] * df['Duration']
    df['Metabolic_Load'] = df['Heart_Rate'] * df['Body_Temp'] * df['Duration']
    df['Age_Duration'] = df['Age'] * df['Duration']
    df['Age_Body_Temp'] = df['Age'] * df['Body_Temp']
    df['Duration_Body_Temp'] = df['Duration'] * df['Body_Temp']
    df['Age_Duration_Temp'] = df['Age'] * df['Duration'] * df['Body_Temp']

    # Height & Weight interactions 
    df['Height_Weight'] = df['Height'] * df['Weight']
    df['Height_Duration'] = df['Height'] * df['Duration']
    df['Weight_Duration'] = df['Weight'] * df['Duration']
    df['Weight_HeartRate'] = df['Weight'] * df['Heart_Rate']
    df['Weight_BodyTemp'] = df['Weight'] * df['Body_Temp']
    df['Height_Temp_Interaction'] = df['Height'] * df['Body_Temp']
    df['Weight_Duration_Temp'] = df['Weight'] * df['Duration'] * df['Body_Temp']
    df['Height_Duration_Temp'] = df['Height'] * df['Duration'] * df['Body_Temp']
    df['Weight_HR_Duration'] = df['Weight'] * df['Heart_Rate'] * df['Duration']
    df['Height_HR_Duration'] = df['Height'] * df['Heart_Rate'] * df['Duration']

    # Advanced exertion interactions
    df['Weight_Intensity_Index'] = df['Weight'] * df['Intensity_Index']
    df['Height_Intensity_Index'] = df['Height'] * df['Intensity_Index']
    df['Weight_HR_Temp_Interaction'] = df['Weight'] * df['HR_Temp_Interaction']
    df['Height_HR_Temp_Interaction'] = df['Height'] * df['HR_Temp_Interaction']

    # Ratio and Normalized Features
    df['HR_per_kg'] = df['Heart_Rate'] / df['Weight']
    df['Duration_per_kg'] = df['Duration'] / df['Weight']
    df['Temp_per_kg'] = df['Body_Temp'] / df['Weight']
    df['HR_per_cm'] = df['Heart_Rate'] / df['Height']
    df['Duration_per_cm'] = df['Duration'] / df['Height']
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2

    # Energy & exertion approximations
    df['Energy_Exerted'] = df['Weight'] * df['Heart_Rate'] * df['Duration'] / 10000
    df['Weighted_Intensity'] = df['Intensity_Index'] * df['Weight']

    # BMR interactions
    df['BMR_HR'] = df['BMR'] * df['Heart_Rate']
    df['BMR_Duration'] = df['BMR'] * df['Duration']
    df['BMR_Temp'] = df['BMR'] * df['Body_Temp']
    df['BMR_Intensity'] = df['BMR'] * df['Intensity_Index']

    # Polynomial and log features
    df['HR_Squared'] = df['Heart_Rate'] ** 2
    df['Duration_Squared'] = df['Duration'] ** 2
    df['Temp_Squared'] = df['Body_Temp'] ** 2
    df['Log_HR'] = np.log1p(df['Heart_Rate'])

    # Sex-based interaction features
    df['Sex_male_HR'] = df['Heart_Rate'] * (df['Sex'] == 1)
    df['Sex_female_HR'] = df['Heart_Rate'] * (df['Sex'] == 0)

    df['Sex_male_Weight'] = df['Weight'] * (df['Sex'] == 1)
    df['Sex_female_Weight'] = df['Weight'] * (df['Sex'] == 0)

    top25 = ['HR_Temp_Interaction','HR_Duration_Interaction','Metabolic_Load','Age_Duration','Age_Body_Temp','Duration_Body_Temp','Age_Duration_Temp','Height_Duration','Weight_Duration','Age', 'Duration','Heart_Rate', 'Sex', 'Weight_Duration_Temp','Height_Duration_Temp', 'Height_HR_Duration', 'Height_Intensity_Index','HR_per_cm','Duration_per_cm','HR_Squared','Duration_Squared','Log_HR','Sex_male_HR','Sex_female_HR','Sex_male_Weight']

    return df[top25]

"""def feature_engineering(features):
    # Intensity index
    features.append(features[4] / features[3])  #7

    # Log transformations 
    features[0] = np.log1p(features[0]) #
    features[5] = np.log1p(features[5])

    # Basal Metabolic Rate
    features.append(
        10 * features[2] + 
        6.25 * features[1] - 
        5 * features[0] + 
        np.where(features[6] == 1, 5, -161)
    ) #8

    # Core interactions
    features.append(features[4] * features[5])                                  #9      #
    features.append(features[4] * features[3])                                  #10     #
    features.append(features[4] * features[5] * features[3])                    #11     #
    features.append(features[0] * features[3])                                  #12     #
    features.append(features[0] * features[5])                                  #13     #
    features.append(features[3] * features[5])                                  #14     #
    features.append(features[0] * features[3] * features[5])                    #15     #

    # Height & Weight interactions 
    features.append(features[1] * features[2])                                  #16
    features.append(features[1] * features[3])                                  #17     #
    features.append(features[2] * features[3])                                  #18     #
    features.append(features[2] * features[4])                                  #19
    features.append(features[2] * features[5])                                  #20
    features.append(features[1] * features[5])                                  #21
    features.append(features[2] * features[3] * features[5])                    #22     #
    features.append(features[1] * features[3] * features[5])                    #23     #
    features.append(features[2] * features[4] * features[3])                    #24
    features.append(features[1] * features[4] * features[3])                    #25     #

    # Advanced exertion interactions
    features.append(features[2] * features[7])                                  #26
    features.append(features[1] * features[7])                                  #27     #
    features.append(features[2] * features[9])                                  #28
    features.append(features[1] * features[9])                                  #29

    # Ratio and Normalized Features
    features.append(features[4] / features[2])                                  #30
    features.append(features[3] / features[2])                                  #31
    features.append(features[5] / features[2])                                  #32
    features.append(features[4] / features[1])                                  #33     #
    features.append(features[3] / features[1])                                  #34     #
    features.append(features[2] / (features[1] / 100) ** 2)                     #35

    # Energy & exertion approximations  
    features.append(features[2] * features[4] * features[3] / 10000)            #36
    features.append(features[7] * features[2])                                  #37

    # BMR interactions  
    features.append(features[8] * features[4])                                  #38
    features.append(features[8] * features[3])                                  #39
    features.append(features[8] * features[5])                                  #40
    features.append(features[8] * features[7])                                  #41

    # Polynomial and log features   
    features.append(features[4] ** 2)                                           #42     #
    features.append(features[3] ** 2)                                           #43     #
    features.append(features[5] ** 2)                                           #44
    features.append(np.log1p(features[4]))                                      #45     #

    # Sex-based interaction features    
    features.append(features[4] * (features[6] == 1))                           #46     #
    features.append(features[4] * (features[6] == 0))                           #47     #

    features.append(features[2] * (features[6] == 1))                           #48     #
    features.append(features[2] * (features[6] == 0))                           #49

    return [features[i] for i in [0,3,4,6,9,10,11,12,13,14,15,17,18,22,23,25,27,33,34,42,43,45,46,47,48]]
"""
#Load the model
model = CatBoostRegressor()
model.load_model("A:\ML_Projects\Predict Calorie Expenditure\App\calories_model_catb.cbm")

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Server is running! Go to /predict for predictions.", 200

@app.route('/predict', methods = ['POST'])
def predict():
    try:
        data = request.json
        print("Request JSON recieved :", data)

        #Collect features to ready for feature engineering
        raw = pd.DataFrame({
            'Age':          data['age'],            #0 #
            'Height':       data['height'],         #1
            'Weight':       data['weight'],         #2
            'Duration':     data['duration'],       #3 #
            'Heart_Rate':   data['heart_rate'],     #4 #
            'Body_Temp':    data['body_temp'],      #5
            'Sex':          data['sex']             #6 #
        }, index = [1])

        features = feature_engineering(raw)

        print(raw)
        print(features)

        prediction = model.predict(features)
        prediction = np.expm1(prediction)

        return jsonify({'Calories' : float(prediction[0])})
    
    except Exception as e:
        print("Error :", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    #flask run app
    app.run(debug = True)