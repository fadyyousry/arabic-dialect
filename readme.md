# Arabic Dialect Identification
## Download models
The application depends on two models SVM and Arabert <br />
You can download them:
* [SVM (model.pkl)](https://www.kaggle.com/fadyelkbeer/ml-arabic-dialects/data)
* [Medium-Arabert (arabert_arabic_dialect.pth)](https://www.kaggle.com/fadyadeeb/fine-tuning-arabert/data?scriptVersionId=90083716)

## Run

run `python app/main.py`

### API
POST request  `http:localhost:5000/predict` <br />
body=` { "text" : ['something in arabic', 'another arabic sentence'] }`
