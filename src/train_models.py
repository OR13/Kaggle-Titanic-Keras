

from tabulate import tabulate

import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.cross_validation import train_test_split, StratifiedKFold

import numpy as np
import copy
from sklearn.cross_validation import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.optimizers import SGD
from keras import backend as K

np.random.seed(1337)  # for reproducibility

model_name = 'keras_model.h5'

def fit(model, train_X, train_y):
    model.fit(train_X, train_y)
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


def write_prediction(name, model, full, test_X):
    test_Y = model.predict(test_X)
    passenger_id = full[891:].PassengerId
    test = pd.DataFrame({'PassengerId': passenger_id, 'Survived': test_Y})
    # test.shape
    # print(test.head())
    test = test.astype(int)
    test.to_csv(name, index=False)



def report_findings(full, titanic, full_X):
    
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

    print("""
    We can evaluate the accuracy of the model by using the validation set where we know the actual outcome.
    This data set have not been used for training the model, so it's completely new to the model.
    We then compare this accuracy score with the accuracy when using the model on the training data.
    If the difference between these are significant this is an indication of overfitting.
    We try to avoid this because it means the model will not generalize well to new data and is expected to perform poorly.
    """)

    #############

    # test_X, full, train_X, train_y, valid_X, valid_y

    model_trainers = [
        train_random_forest,
        train_svc,
        train_gradient_boosting,
        train_k_neighbors,
        train_gaussian_nb,
        train_logistic_regression,
        train_decision_tree
    ]

    model_names = [
        "train_random_forest",
        "train_svc",
        "train_gradient_boosting",
        "train_k_neighbors",
        "train_gaussian_nb",
        "train_logistic_regression",
        "train_decision_tree"
    ]

    def get_trained_models(model_trainers):

        trained_models = {}

        for i in range(0, len(model_names)):
            model_name = model_names[i]
            model_trainer = model_trainers[i]
            model = model_trainer(
                test_X, full, train_X, train_y, valid_X, valid_y)

            trained_models[model_name] = model

        return trained_models

    def get_trained_model_scores(trained_models):

        trained_model_scores = []

        for name, model in trained_models.items():
            train_score = model.score(train_X, train_y)
            test_score = model.score(valid_X, valid_y)
            score = [name, train_score, test_score]
            trained_model_scores.append(score)

        write_prediction(name, model, full, test_X)
        return trained_model_scores

    print()

    trained_models = get_trained_models(model_trainers)

    trained_model_scores = get_trained_model_scores(trained_models)

    for t in trained_model_scores:
        a = float(t[1])
        b = float(t[2])
        c = -1 * 100 * (b - a) / a
        t.append(c)

    trained_model_scores = sorted(
        trained_model_scores, key=lambda score:  score[3])

    print(tabulate(trained_model_scores, headers=[
          "classifier", "train", "test", "difference"]))



def train_and_predict():
    # far too much repetition, just cleaning the data for the neural net

    df = pd.read_csv('./data/full_train.csv')
    train, test = train_test_split(df, test_size=0.15)

    # print(train.head())

    X_train = train.drop(['Survived'], 1)
    X_train = X_train.as_matrix()

    X_test = test.drop(['Survived'], 1)
    X_test = X_test.as_matrix()

    y_train = train['Survived']
    y_test = test['Survived']

    # from class vector to binary matrix
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    nb_classes = 2
    nb_epoch = 128
    batch_size = 64

    model = Sequential()
    model.add(Dense(16, input_shape=X_train.shape[1:]))
    model.add(Activation('tanh'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=batch_size,
              nb_epoch=nb_epoch, validation_data=(X_test, y_test), verbose=2)

    model.save(model_name)

    test_data = pd.read_csv('./data/test_X.csv')
    backup_test = copy.copy(test_data)

    X_test = test_data.as_matrix()

    prediction = model.predict(X_test, batch_size=batch_size, verbose=0)

    def make_bit_array(arr):
        return 1 if arr[0] < arr[1] else 0

    prediction_bitarray = list(map(make_bit_array, prediction))

    # print ('predictions\n', prediction_bitarray)

    passenger_ids = pd.read_csv('./data/test.csv').PassengerId

    df = pd.DataFrame({'PassengerId': passenger_ids,
                       'Survived': prediction_bitarray, })

    # print(df.head())

    df.to_csv('./pred/keras.csv', index=False)

    K.clear_session()
