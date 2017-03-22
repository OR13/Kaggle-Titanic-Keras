from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import pandas as pd


def fit(model, train_X, train_y):
    model.fit(train_X, train_y)
   # print(model.score(train_X, train_y), model.score(valid_X, valid_y))
    return model


def train_random_forest(test_X, full, train_X, train_y, valid_X, valid_y):
    model = RandomForestClassifier(n_estimators=100)
    return fit(model, train_X, train_y)


def train_svc(test_X, full, train_X, train_y, valid_X, valid_y):
    model = SVC()
    return fit(model, train_X, train_y)


def train_gradient_boosting(test_X, full, train_X, train_y, valid_X, valid_y):
    model = GradientBoostingClassifier()
    return fit(model, train_X, train_y)


def train_k_neighbors(test_X, full, train_X, train_y, valid_X, valid_y):
    model = KNeighborsClassifier(n_neighbors=3)
    return model.fit(train_X, train_y)


def train_gaussian_nb(test_X, full, train_X, train_y, valid_X, valid_y):
    model = GaussianNB()
    return fit(model, train_X, train_y)


def train_logistic_regression(test_X, full, train_X, train_y, valid_X, valid_y):
    model = LogisticRegression()
    return fit(model, train_X, train_y)


def train_decision_tree(test_X, full, train_X, train_y, valid_X, valid_y):
    model = DecisionTreeClassifier()
    return fit(model, train_X, train_y)
