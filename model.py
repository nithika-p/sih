
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math

dataset = pd.read_csv('sih_data.csv')
X = dataset.iloc[:, 0:7].values
y = dataset.iloc[:, 7].values
# Encoding categorical data
labelencoder_X_1 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [4])
X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Initialising the ANN
sihmodel = Sequential()

# Adding the input layer and the first hidden layer
sihmodel.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'tanh', input_dim = 11))
sihmodel.add(Dropout(0.2))
# Adding the second hidden layer
sihmodel.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'relu'))
sihmodel.add(Dropout(0.2))
sihmodel.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
# Adding the output layer
sihmodel.add(Dense(units = 1, kernel_initializer = 'uniform'))

# Compiling the ANN
sihmodel.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

# Fitting the ANN to the Training set
sihmodel.fit(X_train, y_train, batch_size = 100, epochs=1000)

y_pred = sihmodel.predict(X_test)

print("----------------------ACTUAL VALUE----------------------------")
print(y_test)
y_pred=y_pred.transpose().astype(int)
print("------------------PREDICTED NO. OF DAYS-------------------")
print(y_pred)
count=np.sum(abs(y_pred-y_test)<=5)
accuracy=count/24000
print("============================================================================================================")
print("                                         Accuracy : "+str(accuracy) +"        ")
print("============================================================================================================")
