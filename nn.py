import numpy as np
import pandas as pd
import copy
from sklearn.cross_validation import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.optimizers import SGD
from keras import backend as K
import pandas as pd

nb_classes = 2
nb_epoch = 100
batch_size = 32

def prep_csv_for_nn(csv_data, dropna=False):
    clean_data = csv_data.drop(['Name', 'Cabin', 'Ticket', 'PassengerId'], axis=1)
    if dropna:
        clean_data = clean_data.dropna()
    clean_data['Sex'] = clean_data['Sex'].map({'male': 0, 'female': 1})
    clean_data['Embarked'] = clean_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    return clean_data
    
def build_model():
    # far too much repetition, just cleaning the data for the neural net
    df = pd.read_csv('./data/train.csv')
    train, test = train_test_split(df, test_size=0.2)

    train = prep_csv_for_nn(train)
    test = prep_csv_for_nn(test)

    X_train = train.drop(['Survived'], 1)
    X_train = X_train.as_matrix()

    X_test = test.drop(['Survived'], 1)
    X_test = X_test.as_matrix()

    y_train = train['Survived']
    y_test = test['Survived']

    # from class vector to binary matrix
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    model = Sequential()
    model.add(Dense(32, input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=batch_size,
              nb_epoch=nb_epoch, validation_data=(X_test, y_test), verbose=2)

    model.save('nn.h5')
    return model


def build_submission(model):

    test_data = pd.read_csv('./data/test.csv')
    backup_test = copy.copy(test_data)

    test_data = prep_csv_for_nn(test_data)

    X_test = test_data.as_matrix()

    prediction = model.predict(X_test, batch_size=batch_size, verbose=0)

    def make_bit_array(arr):
        return 1 if arr[0] > arr[1] else 0
            
    prediction_bitarray = list(map(make_bit_array, prediction))

    csv_input = pd.read_csv('./data/test.csv')
    csv_input['Survived'] = prediction_bitarray
    csv_input.to_csv('./data/prediction.csv', index=False)

try:
    model = load_model('nn.h5')
    print('Loading a model file... now generate a submission')
except:
    model = build_model()

build_submission(model)

K.clear_session()