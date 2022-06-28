#importing libraries
import math
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError

#importing datasets.
train_data = pd.read_csv('trainingset.csv')
test_data = pd.read_csv('testset.csv')

train_data.head()

test_data.head()

train_data.describe()

test_data.describe()

train_data.info()

test_data.info()

#checking for null values
train_data.isnull().sum()

test_data.isnull().sum()

#plotting heatmap
import seaborn as sns
plt.figure(figsize=(14,14))
sns.heatmap(train_data.corr(),annot=True,linecolor ='black', linewidths = 1)

import seaborn as sns
plt.figure(figsize=(14,14))
sns.heatmap(test_data.corr(),annot=True,linecolor ='black', linewidths = 1)

#ploting histogram
plt.hist(train_data['price']);

plt.hist(test_data['price']);

#plotting boxplot
plt.figure(figsize=(15,3))
sns.boxplot(train_data['price'])

plt.figure(figsize=(15,3))
sns.boxplot(test_data['price'])

train_data['price'].describe()

test_data['price'].describe()

#plotting pairpot for training dataset.
sns.pairplot(train_data,diag_kind='kde')

#dropping unimportant features from both dataset.
train_data=train_data.drop(['id','date','zipcode','lat','long','yr_renovated','sqft_lot','condition','yr_built','sqft_lot15'],axis=1)
test_data=test_data.drop(['id','date','zipcode','lat','long','yr_renovated','sqft_lot','condition','yr_built','sqft_lot15'],axis=1)

test_data_price=test_data['price']
test_data=test_data.drop(['price'],axis=1)


y = train_data['price']
X = train_data.drop(['price'],axis=1)

tf_X_train, tf_X_test, tf_y_train, tf_y_test = train_test_split(X, y, train_size=0.7, random_state=1)

from sklearn import preprocessing
tf_X_train=preprocessing.normalize(tf_X_train)
tf_X_test=preprocessing.normalize(tf_X_test)

test_data=preprocessing.normalize(test_data)

sns.pairplot(train_data,diag_kind='kde')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

def HousePredictionModel():
    model=Sequential()
    model.add(Dense(128, activation='relu',input_shape=(tf_X_train[0].shape)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model

import numpy as np
k = 4
num_val_samples = len(tf_X_train)
num_epochs = 30
all_scores = []

model = HousePredictionModel()
history=model.fit(x=tf_X_train,y=tf_y_train, epochs=num_epochs,batch_size=64,verbose=1,validation_data=(tf_X_test,tf_y_test))

ab=model.predict(test_data)
print(ab)

predicted=pd.DataFrame(ab,columns=['Prediction'])

predicted.head()

predicted['original']=test_data_price

predicted.head()





