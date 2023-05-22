# Importing Libraries
import numpy as np # LIBRARY IMPORT FOR LINEAR ALGEBRA
import pandas as pd # LIBRARY IMPORT FOR DATA PROCESSING
from sklearn.model_selection import train_test_split # MODULE IMPORT FOR DATA SPLITTING
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential
from tensorflow.keras.layers import Dense,LSTM, Dropout

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

if __name__ == "__main__": 

    # 1. Load your saved model
    model = keras.models.load_model('./models/Group26_RNN_model.h5')


    # 2. Load your testing data
    #reading the test data
    data_test = pd.read_csv('./data/test_data_RNN.csv')

	#separate features and target
    #the X_train contains only the target data
    # X_train contains the volume, open, high, low values of previous 3 days
    X_test_date = data_test
    X_test = data_test.drop(['Date','target'], axis = 1)
    y_test = data_test['target']

    # MinMaxScalar
    scaler=MinMaxScaler(feature_range=(0,1))
    X_test=scaler.fit_transform(X_test)

    #numpy conversion
    X_test=np.array(X_test)

    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    y_test=np.array(y_test)



    # 3. Run prediction on the test data and output required plot and loss
    y_pred = model.predict(X_test)

    loss_calculation=mean_squared_error(y_test,y_pred)
    print('The loss on test dataset is ',loss_calculation)


    plt.figure(figsize=(20,10))
    plt.plot(y_test, color="green", marker='o', linestyle='dashed', label="real stock price")
    plt.plot(y_pred, color="orange", marker='o', linestyle='dashed', label="predicted stock price")
    plt.title("stock price prediction")
    plt.xlabel("Date(random)")
    plt.ylabel("stock price")
    plt.legend()
    plt.show()
