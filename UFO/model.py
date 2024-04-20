from flask import Flask, request, render_template
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle


model = pickle.load(open('ufo-model.pkl', 'rb'))


country_names = ["Australia", "Canada", "Germany", "United Kingdom", "United States"]


app = Flask(__name__)


def predict_country(seconds, latitude, longitude):
   
    input_data = pd.DataFrame({'Seconds': [seconds], 'Latitude': [latitude], 'Longitude': [longitude]})
    
    predicted_label = model.predict(input_data)[0]
    
    predicted_country = country_names[predicted_label]
    return predicted_country


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
   
    seconds = float(request.form['seconds'])
    latitude = float(request.form['latitude'])
    longitude = float(request.form['longitude'])
    
    predicted_country = predict_country(seconds, latitude, longitude)
    return render_template('result.html', country="Likley Country is : {} ".format(predicted_country))

if __name__ == '__main__':
    app.run(debug=True)
