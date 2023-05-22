# Importing Libraries
import numpy as np # LIBRARY IMPORT FOR LINEAR ALGEBRA
import pandas as pd # LIBRARY IMPORT FOR DATA PROCESSING
from sklearn.model_selection import train_test_split # MODULE IMPORT FOR DATA SPLITTING

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential
from tensorflow.keras.layers import Dense,LSTM, Dropout
import warnings
warnings.filterwarnings("ignore")

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

if __name__ == "__main__": 
    # 1. load your training data
    #Loading Data set
    Dataset = pd.read_csv('./data/q2_dataset.csv')
    type(Dataset)
    list(Dataset.columns)
    Dataset.columns = Dataset.columns.str.strip()
    list(Dataset.columns)
    # we are sorting the data in the ascending order because we are predicting the opening price
    # from the price of the past 3 days
    Dataset['target']= Dataset['Open']
    Dataset['Date'] =pd.to_datetime(Dataset.Date)
    Dataset=Dataset.sort_values(by='Date')
    Dataset.reset_index(inplace=True, drop=True)
    
    column_names=['Open1','High1','Low1','Volume1','Open2','High2','Low2','Volume2','Open3','High3','Low3','Volume3','Date','target']
    df = pd.DataFrame(columns = column_names)
    # Insert data (3 days) + next date opening as target into dataframe
    for i in range(2, len(Dataset)-1):
        d = {"Open1":Dataset.iloc[i-2][3], "High1":Dataset.iloc[i-2][4], "Low1":Dataset.iloc[i-2][5], "Volume1":Dataset.iloc[i-2][2], "Open2":Dataset.iloc[i-1][3], "High2":Dataset.iloc[i-1][4], "Low2":Dataset.iloc[i-1][5], "Volume2":Dataset.iloc[i-1][2], "Open3":Dataset.iloc[i][3], "High3":Dataset.iloc[i][4], "Low3":Dataset.iloc[i][5], "Volume3":Dataset.iloc[i][2], "Date":Dataset.iloc[i+1][0],"target":Dataset.iloc[i+1][3]}
        df = df.append(d, ignore_index=True)
    
    #split the dataset into 70% training and 30% testing
    #train, test = train_test_split(df, test_size=0.30, random_state=0)
    #saveing the train and test data in separate csv files 
    #train.to_csv('./data/train_data_RNN.csv',index=False)
    #test.to_csv('./data/test_data_RNN.csv',index=False)
    # reading the train and test csv
    data_train = pd.read_csv('./data/train_data_RNN.csv')
    data_test = pd.read_csv('./data/test_data_RNN.csv')
    
    #separating features and target
    #the X_train and y_test contains only the target data
    # X_train and X_test contains the volume, open, high, low values of previous 3 days
    X_train = data_train.drop(['Date','target'], axis = 1)
    y_train = data_train['target']
    X_test_date = data_test
    X_test = data_test.drop(['Date','target'], axis = 1)
    y_test = data_test['target']
    
    #scaling the dataset using minmaxscaler
    #We are scalling the data becasue the data is widely varied.
    scaler=MinMaxScaler(feature_range=(0,1))
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    
    #numpy array conversion
    X_train=np.array(X_train)
    X_test=np.array(X_test)
    
    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    
    # Building a Model
    model = Sequential()
    #adding LSTM layer with 50 LSTM units
    model.add(LSTM(50,input_shape=(X_train.shape[1],1),return_sequences=True))
    #adding LSTM layer with 150 LSTM units
    model.add(LSTM(150))
    #adding dense layer
    model.add(Dense(1,activation='linear'))

    #'mean_squared_error' has been used as loss function
    # Optimizer: Here adam optimizer has been used. 
    # Adam is an adaptive learning rate optimization algorithm that’s been designed specifically for
    # training deep neural networks.

    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae'])
    
    model.summary()

    # 2. Train your network
    History = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=750,batch_size=64,verbose=1)
    
    # 3. Save your model
    model.save('./models/Group26_RNN_model.h5')
    