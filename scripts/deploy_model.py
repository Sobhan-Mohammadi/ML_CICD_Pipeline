from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('models/house_price_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = pd.DataFrame(data, index=[0])
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction[0]})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
