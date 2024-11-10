import pandas as pd
import joblib
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Initialize the Flask app
app = Flask(__name__)

# Load data and preprocess
data = pd.read_csv('adult.brl.csv', na_values=" ?")

# Encode categorical features
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split data into features and target
x = data.drop("income", axis=1)
y = data["income"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=63)

# Train the model
model = RandomForestClassifier(n_estimators=150, random_state=63)
model.fit(x_train, y_train)

# Save the model and encoders (optional, you can comment this if not required)
joblib.dump(model, "model.joblib")
joblib.dump(label_encoders, "label_encoders.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Prepare input for prediction
    input_data = pd.DataFrame([data])

    # Encode categorical features for the input DataFrame
    for column, le in label_encoders.items():
        if column in input_data.columns:
            input_data[column] = le.transform(input_data[column])
    
    # Make prediction
    prediction = model.predict(input_data)
    predicted_class = "<=50K" if prediction[0] == 0 else ">50K"
    
    return jsonify({'predicted_income': predicted_class})

@app.route('/', methods=['GET'])
def home():
    return "Income Prediction API is running!"


if __name__ == '__main__':
    app.run(debug=True)