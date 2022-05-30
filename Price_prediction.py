
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


pip install tensorflow

df=pd.read_csv("trainingset.csv")

df.head()

df.isnull().sum()

df.info()

df=df.drop(['id'],axis=1)

df=df.drop(['date'],axis=1)

df

plt.figure(figsize=(12,12))
sns.heatmap(df.corr(),annot=True,linecolor ='black', linewidths = 1)

X=df.iloc[:,1:].values
Y=df.iloc[:,0].values

print(X)

print(Y)

Y = Y.reshape((17290, -1))

print(Y)

Y

X

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X_train=sc_X.fit_transform(X_train)
Y_train=sc_y.fit_transform(Y_train)

print(Y_train)

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

ann.fit(X_train, Y_train, batch_size = 32, epochs = 100)

y_pred = ann.predict(X_test)

print(y_pred)

df1=Y_test.copy()

df1

print(y_pred)

print(np.concatenate((y_pred.reshape(len(y_pred),1), Y_test.reshape(len(Y_test),1)),1))

plt.scatter((Y_test),(y_pred))







