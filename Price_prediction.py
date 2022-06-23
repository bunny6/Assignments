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
train_data = pd.read_csv('trainingset.csv')
test_data = pd.read_csv('testset.csv')
price = 'price'

train_data.head()

test_data.head()

#checking for null values.
train_data.isnull().sum()

test_data.isnull().sum()

#plotting the heatmap for training dataset for checking the correlation between the features.
import seaborn as sns
plt.figure(figsize=(14,14))
sns.heatmap(train_data.corr(),annot=True,linecolor ='black', linewidths = 1)

#plotting the heatmap for test dataset for checking the correlation between the features.
plt.figure(figsize=(14,14))
sns.heatmap(test_data.corr(),annot=True,linecolor ='black', linewidths = 1)

#dropping features which are not having good correlation with the target column.
train_data=train_data.drop(['id','date','zipcode','lat','long','yr_renovated','sqft_lot','condition','yr_built','sqft_lot15'],axis=1)
test_data=test_data.drop(['id','date','zipcode','lat','long','yr_renovated','sqft_lot','condition','yr_built','sqft_lot15'],axis=1)

#plotting heatmap after dropping unnecessary columns for training data.
plt.figure(figsize=(14,14))
sns.heatmap(train_data.corr(),annot=True,linecolor ='black', linewidths = 1)

#plotting heatmap after dropping unnecessary columns for test dataset.
plt.figure(figsize=(14,14))
sns.heatmap(test_data.corr(),annot=True,linecolor ='black', linewidths = 1)

#concating the training and testing datasets.
final_df=pd.concat([train_data,test_data],axis=0)

#spliting the data into train and test..
y = final_df['price'].copy()
X = final_df.drop('price', axis=1).copy()

tf_X_train, tf_X_test, tf_y_train, tf_y_test = train_test_split(X, y, train_size=0.7, random_state=1)

#scaling the data.
from sklearn import preprocessing
tf_X_train=preprocessing.normalize(tf_X_train)
tf_X_test=preprocessing.normalize(tf_X_test)

tf_X_train[0]

#importing  and loading the model.
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

#training the model.
model = HousePredictionModel()
history=model.fit(x=tf_X_train,y=tf_y_train, epochs=num_epochs,batch_size=64,verbose=1,validation_data=(tf_X_test,tf_y_test))

#predicting on the new data.
test_input=[[0.00112569, 0.00093854, 0.5827965 , 0.00112528, 0.        ,
       0.        , 0.00300210, 0.5827943 , 0.        , 0.56628666]]
       
aa=model.predict(test_input)
print(aa)

#predicting on the test dataset.
ab=model.predict(tf_X_test)
print(ab)

ab.shape

final_df1=pd.DataFrame(data=ab,columns=["predicted"])
final_df1.shape

final_df1['Original']=tf_y_test.copy

print(final_df1.head())

print(tf_y_test)






