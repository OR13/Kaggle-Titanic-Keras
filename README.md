
# Kaggle Titanic Problem with Keras

```
conda create --name TitanicEnv
source activate TitanicEnv
pip install git+git://github.com/fchollet/keras.git
pip install tensorflow
pip install tensorflow-gpu
```

### Training Data

You may need to turn:
```
Survived
0
1
1
0
```

into

```
Survived Died
0 1
1 0
1 0
0 1
```

In Excel: =IF(OR(B2,B2="1"),0,1)

https://docs.google.com/spreadsheets/d/1OGfHeiLGkdHXvooDvZMt6yLnhiqyq1I-W569Mdt1UWY/edit#gid=1841073540

### Reading

https://github.com/ChaseByInfinity/titanic-neural-network

https://github.com/Dysvalence/Kaggle-Titanic

https://github.com/agconti/kaggle-titanic

https://vkolachalama.blogspot.com/2016/05/keras-implementation-of-mlp-neural.html