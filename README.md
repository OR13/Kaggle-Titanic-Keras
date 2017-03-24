
# Kaggle Titanic Problem with Keras

```
conda create --name TitanicEnv
source activate TitanicEnv
pip install git+git://github.com/fchollet/keras.git
pip install tensorflow
pip install tensorflow-gpu
```


```

Read More https://www.kaggle.com/seanhgorman/titanic/been-coding-for-a-week-lol

Datasets: full: (1309, 12) titanic: (891, 12)

We can evaluate the accuracy of the model by using the validation set where we know the actual outcome.
This data set have not been used for training the model, so it's completely new to the model.
We then compare this accuracy score with the accuracy when using the model on the training data.
If the difference between these are significant this is an indication of overfitting.
We try to avoid this because it means the model will not generalize well to new data and is expected to perform poorly.

classifier                    train      test    difference
-------------------------  --------  --------  ------------
train_logistic_regression  0.797753  0.781513       2.03574
train_gaussian_nb          0.730337  0.70028        4.11549
train_gradient_boosting    0.911985  0.795518      12.7707
train_k_neighbors          0.829588  0.686275      17.2753
train_svc                  0.840824  0.669468      20.3796
train_random_forest        0.994382  0.767507      22.8157
train_decision_tree        0.994382  0.753501      24.2242
```



### Reading

This shit is awesome.

https://www.kaggle.com/seanhgorman/titanic/been-coding-for-a-week-lol

https://www.kaggle.com/ujjwalkg/titanic/titanic-data-science-solutions

https://github.com/ChaseByInfinity/titanic-neural-network

https://github.com/Dysvalence/Kaggle-Titanic

https://github.com/agconti/kaggle-titanic

https://vkolachalama.blogspot.com/2016/05/keras-implementation-of-mlp-neural.html