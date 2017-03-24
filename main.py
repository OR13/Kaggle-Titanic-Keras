
# From Notebook
# https://www.kaggle.com/oriesteele/titanic/been-coding-for-a-week-lol/editnb


from src.data_processing import load, clean
from src.train_models import report_findings, train_and_predict
# from src.keras_predict import train_and_predict

full, titanic = load()

full_X = clean(full, titanic)

report_findings(full, titanic, full_X)

train_and_predict()

