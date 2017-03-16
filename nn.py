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

model_name = 'model.h5'
nb_classes = 2
nb_epoch = 128
batch_size = 32


import itertools
import operator

def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]


def prep_csv_for_nn(csv_data, for_training=False):
    data = csv_data.drop(
        ['Cabin', 'Ticket', 'PassengerId'], axis=1)

    if for_training:
        data = data.dropna()

    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # All the ages with no data make the median of the data
    data['Age'] = data['Age'].replace(to_replace=[''], value=np.median(data['Age']) )

    # All missing embarks just make them embark from most common place
    data['Embarked'] = data['Embarked'].replace(to_replace=[''], value=most_common(data['Embarked']) )

    # All the missing prices assume median of their respective class
    # for i in xrange(np.size(test_data[0::,0])):
    #     if test_data[i,7] == '':
    #         test_data[i,7] = np.median(test_data[(test_data[0::,7] != '') &\
    #                                             (test_data[0::,0] == test_data[i,0])\
    #             ,7].astype(np.float))

    # Changes from:
    # https://github.com/Dysvalence/Kaggle-Titanic/blob/master/Titanic%20DNN.ipynb

    # Infer Rank From Name
    # Basically makes no difference... or hurts....

    data['Title'] = data['Name'].apply(
        lambda x: ((x.split(', ')[1]).split(' ')[0]))

    rich_ppl = ['Don.', 'Rev.', 'Master.', 'Dr.', 'Col.', 'Capt.',
                'Major.', 'Jonkheer.', 'Lady.', 'the', 'Sir.', 'Dona.']

    poor_ppl = ['Miss.', 'Mlle.', 'Ms.', 'Mrs.', 'Mme.', 'Mr.']

    data['Title'] = data['Title'].replace(to_replace=rich_ppl, value=1)
    data['Title'] = data['Title'].replace(to_replace=poor_ppl, value=0)
    data['Title'] = list(data['Title'])

    del data['Name']

    # data.to_csv('./data/???.csv', index=False)

    return data

def build_model():
    # far too much repetition, just cleaning the data for the neural net
    df = pd.read_csv('./data/train.csv')
    train, test = train_test_split(df, test_size=0.2)

    train = prep_csv_for_nn(train, for_training=True)
    test = prep_csv_for_nn(test, for_training=True)

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

    model.add(Activation('tanh'))
    model.add(Dense(64))

    model.add(Activation('relu'))
    model.add(Dense(128))

    model.add(Activation('relu'))
    model.add(Dense(256))

    model.add(Activation('relu'))
    model.add(Dense(512))

    model.add(Activation('linear'))

    model.add(Dense(2)) # Last dense layer
    model.add(Activation('softmax')) # Softmax activation at the end

    model.compile(loss='mean_squared_error',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=batch_size,
              nb_epoch=nb_epoch, validation_data=(X_test, y_test), verbose=2)

    model.save(model_name)
    return model


def build_submission(model):

    test_data = pd.read_csv('./data/test.csv')
    backup_test = copy.copy(test_data)

    test_data = prep_csv_for_nn(test_data, for_training=False)

    X_test = test_data.as_matrix()

    prediction = model.predict(X_test, batch_size=batch_size, verbose=0)

    def make_bit_array(arr):
        return 1 if arr[0] < arr[1] else 0

    prediction_bitarray = list(map(make_bit_array, prediction))

    csv_input = pd.read_csv('./data/test.csv')
    csv_input = csv_input.drop(['Pclass', 'Name', 'Sex', 'Age', 'SibSp',
                                'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis=1)
    csv_input['Survived'] = prediction_bitarray
    csv_input.to_csv('./data/prediction.csv', index=False)

# model = load_model(model_name)
model = build_model()
build_submission(model)

K.clear_session()
