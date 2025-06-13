from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('california_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        income = float(request.form['MedInc'])
        house_age = float(request.form['HouseAge'])
        avg_rooms = float(request.form['AveRooms'])
        avg_bedrooms = float(request.form['AveBedrms'])
        population = float(request.form['Population'])
        avg_occupancy = float(request.form['AveOccup'])
        latitude = float(request.form['Latitude'])
        longitude = float(request.form['Longitude'])

        # Prepare data and predict
        features = np.array([[income, house_age, avg_rooms, avg_bedrooms, population, avg_occupancy, latitude, longitude]])
        prediction = model.predict(features)[0]

        # Scale up prediction if needed
        predicted_price = f"${round(prediction * 100000, 2)}"  # adjust multiplier if needed

        return render_template('index.html', prediction=predicted_price)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)