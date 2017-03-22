# From Notebook
# https://www.kaggle.com/oriesteele/titanic/been-coding-for-a-week-lol/editnb

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd
import glob
import os
import sys
from itertools import groupby as g


# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# # Modelling Helpers
from sklearn.preprocessing import Imputer, Normalizer, scale
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV

from src.prep import process_raw
from src.predict import write_prediction
from src.train import train_random_forest, train_svc, train_gradient_boosting, train_k_neighbors, train_gaussian_nb, train_logistic_regression, train_decision_tree

from src.keras_predict import train_and_predict
# get titanic & test csv files as a DataFrame
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

full = train.append(test, ignore_index=True)
titanic = full[:891]

del train, test

print('Datasets:', 'full:', full.shape, 'titanic:', titanic.shape)


# Run the code to see the variables, then read the variable description below to understand them.
# print(titanic.head())

# print(titanic.describe())

full_X = process_raw(titanic, full)

# Create all datasets that are necessary to train, validate and test models
train_valid_X = full_X[0:891]
train_valid_y = titanic.Survived
test_X = full_X[891:]
train_X, valid_X, train_y, valid_y = train_test_split(
    train_valid_X, train_valid_y, train_size=.6)

# print (full_X.shape , train_X.shape , valid_X.shape , train_y.shape ,
# valid_y.shape , test_X.shape)

# print(train_valid_y.head())

train_valid_y.to_csv('./data/train_Y.csv', index=False)
train_valid_X.to_csv('./data/train_X.csv', index=False)

full_train = pd.concat([train_valid_X, train_valid_y.astype(int)], axis=1)
full_train.to_csv('./data/full_train.csv', index=False)

test_X.to_csv('./data/test_X.csv', index=False)

#############
random_forest_model = train_random_forest(
    test_X, full, train_X, train_y, valid_X, valid_y)
write_prediction('./pred/random_forest.csv', random_forest_model, full, test_X)

#############
svc_model = train_svc(
    test_X, full, train_X, train_y, valid_X, valid_y)
write_prediction('./pred/svc.csv', svc_model, full, test_X)

#############
gradient_boosting_model = train_gradient_boosting(
    test_X, full, train_X, train_y, valid_X, valid_y)
write_prediction('./pred/gradient_boosting.csv',
                 gradient_boosting_model, full, test_X)


#############
k_neighbors_model = train_k_neighbors(
    test_X, full, train_X, train_y, valid_X, valid_y)
write_prediction('./pred/k_neighbors.csv',
                 k_neighbors_model, full, test_X)


#############
gaussian_nb_model = train_gaussian_nb(
    test_X, full, train_X, train_y, valid_X, valid_y)
write_prediction('./pred/gaussian_nb.csv',
                 gaussian_nb_model, full, test_X)


#############
logistic_regression_model = train_logistic_regression(
    test_X, full, train_X, train_y, valid_X, valid_y)
write_prediction('./pred/logistic_regression.csv',
                 logistic_regression_model, full, test_X)


#############
decision_tree_model = train_logistic_regression(
    test_X, full, train_X, train_y, valid_X, valid_y)
write_prediction('./pred/decision_tree.csv',
                 decision_tree_model, full, test_X)


def merge_predictions():

    prediction_files = glob.glob(os.getcwd() + "/pred/*.csv")

    all_predictions = []

    passenger_ids = pd.read_csv('./data/test.csv').PassengerId

    # print(passenger_ids)

    # all_predictions.append(passenger_ids)

    for prediction_file in prediction_files:
        # print(prediction_file)

        if "all_predictions" in prediction_file:
            continue

        prediction = pd.read_csv(prediction_file)

        name = prediction_file.split('/pred/')[1]
        # print(name)

        prediction = prediction.drop('PassengerId', axis=1)
        prediction = prediction.rename(index=str, columns={"Survived": name})
        prediction = prediction.astype(int)
        all_predictions.append(prediction)

    # print(all_predictions)

    all_preds = pd.concat(all_predictions, axis=1)
    all_preds.to_csv('./pred/all_predictions.csv', index=False)


merge_predictions()


def combine_predictions():

    all_predictions = pd.read_csv('./pred/all_predictions.csv')

    # print(all_predictions.head())

    def most_common(lst):
        return max(set(lst), key=lst.count)

    final_predictions = []

    for index, row in all_predictions.iterrows():

        common_prediction = most_common(list(row))
        # print(common_prediction)
        final_predictions.append(common_prediction)

    passenger_ids = pd.read_csv('./data/test.csv').PassengerId

    df = pd.DataFrame({'Survived': final_predictions})

    combined = pd.concat([passenger_ids, df], axis=1)

    combined.to_csv('./data/combined_predictions.csv', index=False)


combine_predictions()

train_and_predict()