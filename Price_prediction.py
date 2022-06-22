#importing libraries.
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
import seaborn as sns
plt.figure(figsize=(14,14))
sns.heatmap(test_data.corr(),annot=True,linecolor ='black', linewidths = 1)

#dropping the columns which have less correlation with the targer column
train_data=train_data.drop(['id','date','zipcode','lat','long','yr_renovated','sqft_lot','condition','yr_built','sqft_lot15'],axis=1)
test_data=test_data.drop(['id','date','zipcode','lat','long','yr_renovated','sqft_lot','condition','yr_built','sqft_lot15'],axis=1)

#plotting heatmap after dropping unnecessary columns for training data.
plt.figure(figsize=(14,14))
sns.heatmap(train_data.corr(),annot=True,linecolor ='black', linewidths = 1)

#plotting heatmap after dropping unnecessary columns for test dataset.
plt.figure(figsize=(14,14))
sns.heatmap(test_data.corr(),annot=True,linecolor ='black', linewidths = 1)

#concating the dataset.
final_df=pd.concat([train_data,test_data],axis=0)

final_df.head()

#spliting the dataset into X and Y.
y = final_df['price'].copy()
X = final_df.drop('price', axis=1).copy()

#scaling the dataset.
scaler = StandardScaler()

X = scaler.fit_transform(X)

#splitting it into train and test.
tf_X_train, tf_X_test, tf_y_train, tf_y_test = train_test_split(X, y, train_size=0.7, random_state=1)

tf_X_train.shape

#creating model using tensorflow.
inputs = tf.keras.Input(shape=(10,))
hidden = tf.keras.layers.Dense(64, activation='relu')(inputs)
hidden = tf.keras.layers.Dense(64, activation='relu')(hidden)
outputs = tf.keras.layers.Dense(1, activation='linear')(hidden)

tf_model = tf.keras.Model(inputs, outputs)


tf_model.compile(
    optimizer='adam',
    loss='mse'
)

#training the model.
history = tf_model.fit(
    tf_X_train,
    tf_y_train,
    validation_split=0.12,
    batch_size=32,
    epochs=10
)

import numpy as np
tf_rmse = np.sqrt(tf_model.evaluate(tf_X_test, tf_y_test))

print("TensorFlow RMSE:", tf_rmse)

#predicting on the test dataset.
pred_1 = tf_model.predict(tf_X_test)

df_prediction = pd.DataFrame(pred_1, columns=['Prediction'])

df_prediction.head()

# floating point number
float_number = df_prediction['Prediction']
# Convert float to string
string_number = str(float_number)
# Access the first digit using indexing
first_digit = string_number[0]
# Convert first digit string value to integer
first_digit = int(first_digit)
# Print the first digit to console
print(f'The first digit of {float_number} is: {first_digit}')






