from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
base_dir = r"C:\Users\DELL\Desktop\end-to-end-Mental-Health-project\src\Mental_Health\artifacts"

# Load the model and scaler from the specific location
model = pickle.load(open(f"{base_dir}\\logistic_regression.pkl", "rb"))
scaler = pickle.load(open(f"{base_dir}\\scaler.pickle", "rb"))

# Ensure exact feature order as used in training
feature_order = ['Gender', 'Age', 'Work Pressure', 'Job Satisfaction', 'Dietary Habits',
                 'Have you ever had suicidal thoughts ?', 'Work Hours', 'Financial Stress',
                 'Family History of Mental Illness']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract data from form
        features = {
            "Age": int(request.form["age"]),
            "Gender": int(request.form["gender"]),
            "Work Pressure": int(request.form["work_pressure"]),
            "Job Satisfaction": int(request.form["job_satisfaction"]),
            "Dietary Habits": int(request.form["dietary_habits"]),
            "Have you ever had suicidal thoughts ?": int(request.form["suicidal_thoughts"]),
            "Work Hours": int(request.form["work_hours"]),
            "Financial Stress": int(request.form["financial_stress"]),
            "Family History of Mental Illness": int(request.form["family_history"])
        }

        # Convert dictionary to DataFrame **with correct column order**
        input_features = pd.DataFrame([[features[col] for col in feature_order]], columns=feature_order)

        # Scale the input features
        scaled_features = scaler.transform(input_features)

        # Make prediction
        prediction = model.predict(scaled_features)

        # Map prediction to result and messages
        if prediction[0] == 1:
            message = "Your mental health may be affected. Please consider seeking professional help and taking care of yourself."
        else:
            message = "Your mental health seems to be in a good state. Keep maintaining a balanced lifestyle!"

        return render_template('result.html', prediction_text=message)

if __name__ == "__main__":
    app.run(debug=True)
