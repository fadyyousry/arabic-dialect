from flask import Flask, request, jsonify
import joblib
import pandas as pd
from helper import *

print('load ML ...')
clf = joblib.load('../models/model_ml.pkl')
print('done! ML')

print('load DL ...')
model = load_dl()
print('done! DL')

print('load tokenizer ...')
tokenizer = load_tokenizer()
print('done! tokenizer')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_ = request.json
        query = pd.DataFrame.from_dict(json_)
        # prediction = clf.predict(query['text'])
        preds = predict_dialect(query, model, tokenizer)
        return jsonify({
            'ML prediction': list(prediction),
            'DL prediction': preds
            })
    except:
        return jsonify({'error': 'error during prediction'})

if __name__=="__main__":
    print('start .. ')
    app.run(debug=False)