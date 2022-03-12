from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

app.route('/predict', methods=['POST'])
def predict():
    try:
        json_ = request.json
        query = pd.DataFrame(json_)
        prediction = clf.predict(query)
        return jsonify({'prediction': list(prediction)})
    except:
        return jsonify({'error': 'error during prediction'})

if __name__ == '__main__':
    clf = joblib.load('models/model.pkl')