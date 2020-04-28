'''
 Extract only the feature that should be supplied to neural network 
 using keras library which is an high level api of tensor flow for building and trainig deep 
 learning models

'''
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# preprocessed data import
from trainingOpen import x_train, y_train


# rnn call
regressor = Sequential()

# Trainig the Neural Network

# first LSTM Layer
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1],1)))
regressor.add(Dropout(0.2)) # excluding 0.2 percent of data for further input


# second LSTM Layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2)) # excluding 0.2 percent of data for further input


# third LSTM Layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2)) # excluding 0.2 percent of data for further input


# fourth LSTM Layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2)) # excluding 0.2 percent of data for further input


# output layer of Dense, through which we can alter the dimension of output vector
regressor.add(Dense(units = 1)) # only one ouput required so units =1 


# Using Adam(adaptive movement estimation) optimizer 

regressor.compile(optimizer = 'adam', loss='mean_squared_error')
regressor.fit(x_train,y_train, epochs=100, batch_size=32)

# batch size  = 32 , i,e 32 training sets utilized for a single iteration
