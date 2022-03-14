# Arabic Dialect Identification
## Download models
The application depends on two models SVM and Arabert
You can download them:
* [SVM (model.pkl)](https://www.kaggle.com/fadyelkbeer/ml-arabic-dialects/data)
* [Medium-Arabert (arabert_arabic_dialect.pth)](https://www.kaggle.com/fadyelkbeer/ml-arabic-dialects/data)

## Run

run `python app/main.py`

### API
POST request  `http:localhost:5000/predict`
body=` { "text" : ['something in arabic', 'another arabic sentence'] }`