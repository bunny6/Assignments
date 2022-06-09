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

#imporing the datasets
train_data = pd.read_csv('trainingset.csv')
test_data = pd.read_csv('testset.csv')

train_data.head()

test_data.head()

#checking for null values.
train_data.isnull().sum()

test_data.isnull().sum()

#ploting the heatmap for checking the correlation between the target column and other features
import seaborn as sns
plt.figure(figsize=(14,14))
sns.heatmap(train_data.corr(),annot=True,linecolor ='black', linewidths = 1)

plt.figure(figsize=(14,14))
sns.heatmap(test_data.corr(),annot=True,linecolor ='black', linewidths = 1)

#droping unwanted columns
train_data=train_data.drop(['id','date','zipcode','lat','long','yr_renovated','sqft_lot','condition','yr_built','sqft_lot15'],axis=1)
test_data=test_data.drop(['id','date','zipcode','lat','long','yr_renovated','sqft_lot','condition','yr_built','sqft_lot15'],axis=1)

#spliting the dataset
X_train=train_data.iloc[:,1:].values
Y_train=train_data.iloc[:,0].values

X_test=test_data.iloc[:,1:].values
Y_test=test_data.iloc[:,0].values

Y_train = Y_train.reshape((17290, -1))
Y_test = Y_test.reshape((4323, -1))

#scaling the dataset.
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X_train=sc_X.fit_transform(X_train)
Y_train=sc_y.fit_transform(Y_train)
X_test=sc_X.fit_transform(X_test)
Y_test=sc_y.fit_transform(Y_test)

#importing the ann.
ann = tf.keras.models.Sequential()

#adding  3 hidden layers
ann.add(tf.keras.layers.Dense(units=160, activation='relu'))

ann.add(tf.keras.layers.Dense(units=480, activation='relu'))

ann.add(tf.keras.layers.Dense(units=160, activation='relu'))
 
#adding output layer
ann.add(tf.keras.layers.Dense(units=1, activation='linear'))

#loss function
msle=MeanSquaredLogarithmicError()

ann.compile(optimizer = 'adam', loss = 'msle', metrics = ['msle'])

#training the ann 
ANN=ann.fit(X_train, Y_train, batch_size = 64, epochs = 35, validation_split=0.2)

#predicting on train dataset
X_test['prediction'] =  ann.predict(X_test)







