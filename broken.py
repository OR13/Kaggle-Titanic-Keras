import csv
import numpy
import pandas


#import Keras 
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K


dfTrain=pandas.read_csv('./data/train.csv')
dfTest=pandas.read_csv('./data/test.csv')

print('Training Examples:',len(dfTrain))
print('Test Examples:',len(dfTest))
dfAll=dfTrain.append(dfTest) 
print('All Examples:',len(dfAll))

#Determine social ranking from names to create an additional feature
dfAll['Title']=dfAll['Name'].apply( lambda x:( (x.split(', ')[1]).split(' ')[0] ) )

dfAll['Title']=dfAll['Title'].replace(to_replace=['Don.','Rev.','Master.','Dr.','Col.','Capt.',
                                                      'Major.','Jonkheer.','Lady.','the','Sir.','Dona.'], value=1)                                          
dfAll['Title']=dfAll['Title'].replace(to_replace=['Miss.','Mlle.','Ms.','Mrs.','Mme.','Mr.'],value=0)



#Simple imputation, based on this analysis: https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic
dfAll['Fare']=dfAll['Fare'].replace(to_replace=[dfTest['Fare'][1044-892]],value=8.05)
#Numpy does not allow direct comparison to np.nan. This is an admittedly ugly workaround.
NAN=(dfTrain['Embarked'][829])
dfAll['Embarked']=dfTrain['Embarked'].replace(to_replace=[NAN],value='C')

#one-hot encoding
dfAll=pandas.get_dummies(dfAll, columns=['Sex'])
dfAll=pandas.get_dummies(dfAll, columns=['Embarked'])



#Normalization
def normalize(df,strlab):
    df[strlab]=(df[strlab].map(lambda x: float(float(float(x) - float(numpy.min(df[strlab]))) /
                                                     float(float(numpy.max(df[strlab])) - float(numpy.min(df[strlab]))))))
normalize(dfAll,'Age')
normalize(dfAll,'Parch')
normalize(dfAll,'Pclass')
normalize(dfAll,'Fare')
normalize(dfAll,'SibSp')


#Prepare to imputate age.
dfNoAge=dfAll[numpy.isnan(dfAll['Age'])]
dfAge=dfAll[(numpy.isnan(dfAll['Age']))==False]
print('Has age:',len(dfAge))
print('Missing age:',len(dfNoAge))

noAgeAE=dfAll.drop(['Age','Cabin','PassengerId','Name','Survived','Ticket'],axis=1).as_matrix()
ageTrainX=dfAge.drop(['Age','Cabin','PassengerId','Name','Survived','Ticket'],axis=1).as_matrix()
ageTrainY=dfAge.as_matrix(columns=['Age'])


batch_size = 20
nb_classes = 2
nb_epochs = 20
models=[]


models=[]

#Runs multiple models with increasing epochs, and with the option to pick an earlier one.
#Not optimal, but helps deal with bugs and latency in jupyter notebook. Verbosity is disabled for this reason.
for x in range(0,3):
    nb_epoch=nb_epochs*x
    print(nb_epoch)
    model = Sequential()
    model.add(Dense(9, input_shape=(10,)))
    model.add(Activation('tanh'))
    model.add(Dense(10))
    model.add(Activation('linear'))

    model.compile(loss='mean_absolute_error',
              optimizer=RMSprop(),
              metrics=['accuracy'])

    history = model.fit(noAgeAE, noAgeAE,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=0, validation_data=(noAgeAE, noAgeAE))

    score = model.evaluate(noAgeAE, noAgeAE, verbose=1)
    models.append(model)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    print('')
print('\nDone')


# AELayer=model #Selects last model


# batch_size = 1
# nb_classes = 2
# nb_epochs = 20
# models=[]

# #Massive bug here in which each run is not independent, and the initialized layers get backproped over on every run.
# #However, this seems to increase performance, and may be worth looking into.

# for x in range(0,3):
#     nb_epoch=nb_epochs*x
#     print(nb_epoch)
#     model = Sequential()
#     model.add(AELayer.layers[0])
#     model.add(AELayer.layers[1])
#     model.add(Dense(10))
#     model.add(Activation('relu'))
#     model.add(Dense(1))
#     model.add(Activation('linear'))

#     model.compile(loss='mean_squared_error',
#               optimizer=RMSprop(),
#               metrics=['accuracy'])

#     history = model.fit(ageTrainX, ageTrainY,
#                     batch_size=batch_size, nb_epoch=nb_epoch,
#                     verbose=0, validation_data=(ageTrainX, ageTrainY))

#     score = model.evaluate(ageTrainX, ageTrainY, verbose=1)
#     models.append(model)
#     print('Test score:', score[0])
#     print('Test accuracy:', score[1])
#     print('')
# print('\nDone')


# finmod=model #pick last model


# #imputate age from NN model

# dfNoAge=dfNoAge.drop(['Age'],axis=1)
# matNoAge=dfNoAge.drop(['Cabin','PassengerId','Name','Survived','Ticket'], axis=1)

# ages=finmod.predict(matNoAge.as_matrix(),batch_size=1)

# dfNoAge['Age']=ages



# #Recombine data, and resplit to build the classifier
# dfRecombined=dfNoAge.append(dfAge)
# dfRecombined=dfRecombined.drop(['Cabin','Name','Ticket'],axis=1)
# MatAEClass=dfRecombined.drop(['Survived','PassengerId'],axis=1).as_matrix()

# dfRecTest=dfRecombined[numpy.isnan(dfRecombined['Survived'])]
# dfRecTest=dfRecTest.drop(['Survived'],axis=1)
# dfRecTrain=dfRecombined[numpy.isnan(dfRecombined['Survived'])==False]



# """
# I tried splitting off 91 examples for validation but training accuracy was hit pretty hard,
# and validation accuracy was extremely erratic depending on how the random sampling went.
# Since nothing is going right I decided to risk deliberately overfitting the test set
# and decided to initialize the first layer of a deep neural net by running an AE on all the data.
# This might be a viable way to use data with missing labels.
# """

# batch_size = 1
# nb_classes = 2
# nb_epochs = 20
# models=[]

# for x in range(0,3):
#     nb_epoch=nb_epochs*x
#     print(nb_epoch)
#     model = Sequential()
#     model.add(Dense(7, input_shape=(11,)))
#     model.add(Activation('relu'))
#     model.add(Dense(11))
#     model.add(Activation('linear'))

#     model.compile(loss='mean_squared_error',
#               optimizer=RMSprop(),
#               metrics=['accuracy'])

# #Verbosity disabled since it triggers a juptyer notebook bug and crashes the training

#     history = model.fit(MatAEClass, MatAEClass,
#                     batch_size=batch_size, nb_epoch=nb_epoch,
#                     verbose=0, validation_data=(MatAEClass, MatAEClass))

#     score = model.evaluate(MatAEClass, MatAEClass, verbose=1)
#     models.append(model)
#     print('Test score:', score[0])
#     print('Test accuracy:', score[1])
#     print('')
# print('\nDone')

# AEModel=model #Pick model


# #Reseperate training and test data; format labels
# finTrainX=dfRecTrain.drop(['Survived','PassengerId'],axis=1).as_matrix()
# dfRecTrain=pandas.get_dummies(dfRecTrain, columns=['Survived'])
# finTrainY=dfRecTrain.as_matrix(columns=['Survived_0.0','Survived_1.0'])


# """
# Again, theres the massive bug where the first hidden layer is shared by, and subsequently trained by, all the models.
# However this has given me the best test accuracy so far, and fixing this bug drops accuracy by a few percent at least.
# Given that deep nets tend to have vanishing gradients further up this may be a way to counter that, similar to how AE based
# deep belief networks are used.

# I've messed around a lot with varying layers, hidden units, activations, optimizers, loss functions, dropout 
# and regularization, and while the AE initialization helped, getting further with just NN/DNN architectures will
# require either luck or experience. Other public submissions further up the leaderboard have used genetic algorithms and
# multiple models in a committee. Integrating elements of this approach may be helpful.
# """
# batch_size = 1
# nb_classes = 2
# nb_epochs = 10 
# models=[]

# for x in range(0,20):
#     nb_epoch=nb_epochs*x
#     print(nb_epoch)
#     model = Sequential()
#     model.add(AEModel.layers[0])
#     model.add(AEModel.layers[1])
#     #model.add(Dense(100, input_shape=(11,)))
#     #model.add(Activation('relu'))
#     #model.add(Dropout(.2))
#     model.add(Dense(10))
#     model.add(Activation('relu'))
#     #model.add(Dropout(.2))
#     model.add(Dense(2))
#     model.add(Activation('softmax'))

#     model.compile(loss='categorical_crossentropy',
#               optimizer=Adam(),
#               metrics=['accuracy'])

# #Verbosity disabled since it triggers a juptyer notebook bug and crashes the training

#     history = model.fit(finTrainX, finTrainY,
#                     batch_size=batch_size, nb_epoch=nb_epoch,
#                     verbose=0, validation_data=(finTrainX, finTrainY))

#     score = model.evaluate(finTrainX, finTrainY, verbose=1)
#     models.append(model)
#     print('Test score:', score[0])
#     print('Test accuracy:', score[1])
#     print('')
# print('\nDone')


K.clear_session()
