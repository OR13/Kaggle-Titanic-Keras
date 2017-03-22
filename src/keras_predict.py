import numpy as np
import pandas as pd
import copy
from sklearn.cross_validation import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.optimizers import SGD
from keras import backend as K

np.random.seed(1337)  # for reproducibility

model_name = 'keras_model.h5'

def train_and_predict():
    # far too much repetition, just cleaning the data for the neural net

    df = pd.read_csv('./data/full_train.csv')
    train, test = train_test_split(df, test_size=0.7)

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
    nb_epoch = 200
    batch_size = 32

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
