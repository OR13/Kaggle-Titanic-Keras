
# Kaggle Titanic Problem with Keras

```
conda create --name TitanicEnv
source activate TitanicEnv
pip install git+git://github.com/fchollet/keras.git
pip install tensorflow --ignore-installed
pip install tensorflow-gpu --ignore-installed
```

### Reading

This shit is awesome.

https://www.kaggle.com/seanhgorman/titanic/been-coding-for-a-week-lol

https://www.kaggle.com/ujjwalkg/titanic/titanic-data-science-solutions

https://github.com/ChaseByInfinity/titanic-neural-network

https://github.com/Dysvalence/Kaggle-Titanic

https://github.com/agconti/kaggle-titanic

https://vkolachalama.blogspot.com/2016/05/keras-implementation-of-mlp-neural.html



### Sample Run

```
$ python main.py

Using TensorFlow backend.
Datasets: full: (1309, 12) titanic: (891, 12)

    We can evaluate the accuracy of the model by using the validation set where we know the actual outcome.
    This data set have not been used for training the model, so it's completely new to the model.
    We then compare this accuracy score with the accuracy when using the model on the training data.
    If the difference between these are significant this is an indication of overfitting.
    We try to avoid this because it means the model will not generalize well to new data and is expected to perform poorly.
    

classifier                    train      test    difference
-------------------------  --------  --------  ------------
train_logistic_regression  0.814607  0.823529    -1.09533
train_gaussian_nb          0.801498  0.80112      0.0471217
train_gradient_boosting    0.90824   0.815126    10.2521
train_random_forest        0.955056  0.767507    19.6375
train_k_neighbors          0.844569  0.663866    21.396
train_decision_tree        0.955056  0.733894    23.157
train_svc                  0.897004  0.67507     24.7417

Train on 846 samples, validate on 45 samples
Epoch 1/100
0s - loss: 0.6213 - acc: 0.6631 - val_loss: 0.5707 - val_acc: 0.7333
Epoch 2/100
0s - loss: 0.5863 - acc: 0.6962 - val_loss: 0.5679 - val_acc: 0.6889
Epoch 3/100
0s - loss: 0.4985 - acc: 0.7730 - val_loss: 0.4799 - val_acc: 0.8444
Epoch 4/100
0s - loss: 0.4900 - acc: 0.7896 - val_loss: 0.4324 - val_acc: 0.8667
Epoch 5/100
0s - loss: 0.4527 - acc: 0.8097 - val_loss: 0.3949 - val_acc: 0.8444
Epoch 6/100
0s - loss: 0.4527 - acc: 0.8109 - val_loss: 0.4074 - val_acc: 0.8444
Epoch 7/100
0s - loss: 0.4615 - acc: 0.8026 - val_loss: 0.3895 - val_acc: 0.8667
Epoch 8/100
0s - loss: 0.4383 - acc: 0.8168 - val_loss: 0.4002 - val_acc: 0.8667
Epoch 9/100
0s - loss: 0.4360 - acc: 0.8121 - val_loss: 0.4509 - val_acc: 0.8444
Epoch 10/100
0s - loss: 0.4480 - acc: 0.7991 - val_loss: 0.3897 - val_acc: 0.8667
Epoch 11/100
0s - loss: 0.4311 - acc: 0.8144 - val_loss: 0.3825 - val_acc: 0.8667
Epoch 12/100
0s - loss: 0.4259 - acc: 0.8168 - val_loss: 0.4087 - val_acc: 0.7778
Epoch 13/100
0s - loss: 0.4210 - acc: 0.8156 - val_loss: 0.3723 - val_acc: 0.8667
Epoch 14/100
0s - loss: 0.4233 - acc: 0.8132 - val_loss: 0.4171 - val_acc: 0.8444
Epoch 15/100
0s - loss: 0.4205 - acc: 0.8251 - val_loss: 0.3986 - val_acc: 0.8444
Epoch 16/100
0s - loss: 0.4213 - acc: 0.8191 - val_loss: 0.4608 - val_acc: 0.7556
Epoch 17/100
0s - loss: 0.4143 - acc: 0.8180 - val_loss: 0.4293 - val_acc: 0.8444
Epoch 18/100
0s - loss: 0.4168 - acc: 0.8168 - val_loss: 0.4276 - val_acc: 0.8222
Epoch 19/100
0s - loss: 0.4166 - acc: 0.8262 - val_loss: 0.4426 - val_acc: 0.7333
Epoch 20/100
0s - loss: 0.4042 - acc: 0.8203 - val_loss: 0.4015 - val_acc: 0.8444
Epoch 21/100
0s - loss: 0.4058 - acc: 0.8191 - val_loss: 0.4214 - val_acc: 0.8222
Epoch 22/100
0s - loss: 0.4046 - acc: 0.8251 - val_loss: 0.4379 - val_acc: 0.7556
Epoch 23/100
0s - loss: 0.4002 - acc: 0.8274 - val_loss: 0.4092 - val_acc: 0.7556
Epoch 24/100
0s - loss: 0.3933 - acc: 0.8416 - val_loss: 0.4110 - val_acc: 0.8222
Epoch 25/100
0s - loss: 0.3893 - acc: 0.8333 - val_loss: 0.4245 - val_acc: 0.7556
Epoch 26/100
0s - loss: 0.3962 - acc: 0.8333 - val_loss: 0.3840 - val_acc: 0.8444
Epoch 27/100
0s - loss: 0.3903 - acc: 0.8333 - val_loss: 0.3923 - val_acc: 0.8222
Epoch 28/100
0s - loss: 0.3940 - acc: 0.8298 - val_loss: 0.4228 - val_acc: 0.7556
Epoch 29/100
0s - loss: 0.3847 - acc: 0.8298 - val_loss: 0.3698 - val_acc: 0.8444
Epoch 30/100
0s - loss: 0.3808 - acc: 0.8475 - val_loss: 0.3945 - val_acc: 0.8222
Epoch 31/100
0s - loss: 0.3852 - acc: 0.8298 - val_loss: 0.3766 - val_acc: 0.8444
Epoch 32/100
0s - loss: 0.3814 - acc: 0.8404 - val_loss: 0.3819 - val_acc: 0.8222
Epoch 33/100
0s - loss: 0.3801 - acc: 0.8333 - val_loss: 0.3895 - val_acc: 0.8000
Epoch 34/100
0s - loss: 0.3833 - acc: 0.8392 - val_loss: 0.3762 - val_acc: 0.8444
Epoch 35/100
0s - loss: 0.3760 - acc: 0.8369 - val_loss: 0.3844 - val_acc: 0.8000
Epoch 36/100
0s - loss: 0.3716 - acc: 0.8452 - val_loss: 0.4017 - val_acc: 0.7556
Epoch 37/100
0s - loss: 0.3734 - acc: 0.8333 - val_loss: 0.3892 - val_acc: 0.8222
Epoch 38/100
0s - loss: 0.3727 - acc: 0.8428 - val_loss: 0.3704 - val_acc: 0.8444
Epoch 39/100
0s - loss: 0.3791 - acc: 0.8310 - val_loss: 0.3554 - val_acc: 0.8667
Epoch 40/100
0s - loss: 0.3798 - acc: 0.8522 - val_loss: 0.3415 - val_acc: 0.8667
Epoch 41/100
0s - loss: 0.3691 - acc: 0.8404 - val_loss: 0.4059 - val_acc: 0.7333
Epoch 42/100
0s - loss: 0.3736 - acc: 0.8333 - val_loss: 0.3551 - val_acc: 0.8444
Epoch 43/100
0s - loss: 0.3619 - acc: 0.8452 - val_loss: 0.3915 - val_acc: 0.8222
Epoch 44/100
0s - loss: 0.3690 - acc: 0.8322 - val_loss: 0.3817 - val_acc: 0.8000
Epoch 45/100
0s - loss: 0.3628 - acc: 0.8534 - val_loss: 0.3971 - val_acc: 0.8000
Epoch 46/100
0s - loss: 0.3598 - acc: 0.8452 - val_loss: 0.4051 - val_acc: 0.8444
Epoch 47/100
0s - loss: 0.3637 - acc: 0.8440 - val_loss: 0.3549 - val_acc: 0.8667
Epoch 48/100
0s - loss: 0.3670 - acc: 0.8440 - val_loss: 0.3625 - val_acc: 0.8667
Epoch 49/100
0s - loss: 0.3614 - acc: 0.8522 - val_loss: 0.3841 - val_acc: 0.8222
Epoch 50/100
0s - loss: 0.3577 - acc: 0.8452 - val_loss: 0.3908 - val_acc: 0.7556
Epoch 51/100
0s - loss: 0.3510 - acc: 0.8475 - val_loss: 0.3554 - val_acc: 0.8444
Epoch 52/100
0s - loss: 0.3551 - acc: 0.8440 - val_loss: 0.3640 - val_acc: 0.8444
Epoch 53/100
0s - loss: 0.3627 - acc: 0.8404 - val_loss: 0.3834 - val_acc: 0.8222
Epoch 54/100
0s - loss: 0.3527 - acc: 0.8558 - val_loss: 0.3547 - val_acc: 0.8444
Epoch 55/100
0s - loss: 0.3502 - acc: 0.8416 - val_loss: 0.3591 - val_acc: 0.8444
Epoch 56/100
0s - loss: 0.3532 - acc: 0.8534 - val_loss: 0.3928 - val_acc: 0.8000
Epoch 57/100
0s - loss: 0.3458 - acc: 0.8475 - val_loss: 0.3607 - val_acc: 0.8444
Epoch 58/100
0s - loss: 0.3515 - acc: 0.8499 - val_loss: 0.3644 - val_acc: 0.8444
Epoch 59/100
0s - loss: 0.3516 - acc: 0.8522 - val_loss: 0.4031 - val_acc: 0.8222
Epoch 60/100
0s - loss: 0.3509 - acc: 0.8534 - val_loss: 0.3853 - val_acc: 0.8444
Epoch 61/100
0s - loss: 0.3677 - acc: 0.8463 - val_loss: 0.3920 - val_acc: 0.8222
Epoch 62/100
0s - loss: 0.3504 - acc: 0.8440 - val_loss: 0.3714 - val_acc: 0.8444
Epoch 63/100
0s - loss: 0.3423 - acc: 0.8570 - val_loss: 0.3828 - val_acc: 0.8222
Epoch 64/100
0s - loss: 0.3459 - acc: 0.8463 - val_loss: 0.3971 - val_acc: 0.8222
Epoch 65/100
0s - loss: 0.3507 - acc: 0.8511 - val_loss: 0.3492 - val_acc: 0.8667
Epoch 66/100
0s - loss: 0.3447 - acc: 0.8475 - val_loss: 0.3486 - val_acc: 0.8444
Epoch 67/100
0s - loss: 0.3518 - acc: 0.8475 - val_loss: 0.3721 - val_acc: 0.8444
Epoch 68/100
0s - loss: 0.3420 - acc: 0.8522 - val_loss: 0.3686 - val_acc: 0.8222
Epoch 69/100
0s - loss: 0.3515 - acc: 0.8440 - val_loss: 0.3970 - val_acc: 0.8222
Epoch 70/100
0s - loss: 0.3487 - acc: 0.8452 - val_loss: 0.3755 - val_acc: 0.8222
Epoch 71/100
0s - loss: 0.3341 - acc: 0.8511 - val_loss: 0.3693 - val_acc: 0.8222
Epoch 72/100
0s - loss: 0.3409 - acc: 0.8463 - val_loss: 0.3278 - val_acc: 0.8444
Epoch 73/100
0s - loss: 0.3658 - acc: 0.8416 - val_loss: 0.3724 - val_acc: 0.8444
Epoch 74/100
0s - loss: 0.3445 - acc: 0.8416 - val_loss: 0.3604 - val_acc: 0.8444
Epoch 75/100
0s - loss: 0.3414 - acc: 0.8440 - val_loss: 0.3897 - val_acc: 0.8222
Epoch 76/100
0s - loss: 0.3345 - acc: 0.8487 - val_loss: 0.3797 - val_acc: 0.8667
Epoch 77/100
0s - loss: 0.3465 - acc: 0.8499 - val_loss: 0.3998 - val_acc: 0.8667
Epoch 78/100
0s - loss: 0.3475 - acc: 0.8546 - val_loss: 0.3880 - val_acc: 0.8444
Epoch 79/100
0s - loss: 0.3428 - acc: 0.8487 - val_loss: 0.4121 - val_acc: 0.8444
Epoch 80/100
0s - loss: 0.3513 - acc: 0.8416 - val_loss: 0.3846 - val_acc: 0.8222
Epoch 81/100
0s - loss: 0.3527 - acc: 0.8392 - val_loss: 0.3783 - val_acc: 0.8667
Epoch 82/100
0s - loss: 0.3380 - acc: 0.8475 - val_loss: 0.4389 - val_acc: 0.7556
Epoch 83/100
0s - loss: 0.3580 - acc: 0.8333 - val_loss: 0.3680 - val_acc: 0.8444
Epoch 84/100
0s - loss: 0.3312 - acc: 0.8522 - val_loss: 0.3746 - val_acc: 0.8444
Epoch 85/100
0s - loss: 0.3297 - acc: 0.8487 - val_loss: 0.4106 - val_acc: 0.7556
Epoch 86/100
0s - loss: 0.3253 - acc: 0.8617 - val_loss: 0.4110 - val_acc: 0.8000
Epoch 87/100
0s - loss: 0.3383 - acc: 0.8546 - val_loss: 0.3645 - val_acc: 0.8667
Epoch 88/100
0s - loss: 0.3299 - acc: 0.8546 - val_loss: 0.4247 - val_acc: 0.8000
Epoch 89/100
0s - loss: 0.3271 - acc: 0.8570 - val_loss: 0.5183 - val_acc: 0.7111
Epoch 90/100
0s - loss: 0.3382 - acc: 0.8570 - val_loss: 0.4046 - val_acc: 0.8222
Epoch 91/100
0s - loss: 0.3284 - acc: 0.8605 - val_loss: 0.3762 - val_acc: 0.8222
Epoch 92/100
0s - loss: 0.3290 - acc: 0.8558 - val_loss: 0.4213 - val_acc: 0.8222
Epoch 93/100
0s - loss: 0.3399 - acc: 0.8534 - val_loss: 0.3875 - val_acc: 0.8444
Epoch 94/100
0s - loss: 0.3395 - acc: 0.8511 - val_loss: 0.3998 - val_acc: 0.8222
Epoch 95/100
0s - loss: 0.3281 - acc: 0.8570 - val_loss: 0.3889 - val_acc: 0.8222
Epoch 96/100
0s - loss: 0.3292 - acc: 0.8605 - val_loss: 0.3930 - val_acc: 0.8000
Epoch 97/100
0s - loss: 0.3385 - acc: 0.8499 - val_loss: 0.3650 - val_acc: 0.8444
Epoch 98/100
0s - loss: 0.3240 - acc: 0.8534 - val_loss: 0.3658 - val_acc: 0.8222
Epoch 99/100
0s - loss: 0.3196 - acc: 0.8629 - val_loss: 0.3978 - val_acc: 0.8000
Epoch 100/100
0s - loss: 0.3221 - acc: 0.8570 - val_loss: 0.3925 - val_acc: 0.8444
```



