'''


'''
import pandas as pd
import numpy as np 

from ProcessedDataframe import testData, trainData

test_df = testData()
train_df = trainData()

real_stock_price = test_df.iloc[:, 1:2].values # open data

# predicting the stock price of 2017

dataset_total = pd.concat((train_df['Open'], test_df['Open']), axis = 0) # combinig test and train data
inputs = dataset_total[len(dataset_total)- len(test_df) - 60:].values # format the data for input
inputs = inputs.reshape(-1,1) # reshaping

# import of sklearn preprossed minmax  data model to tranform the input
from trainingOpen import sc
inputs = sc.transform(inputs) # range(0,1)

# form test data with timestap 60 
x_test = []

for i in range(60, 80):
    x_test.append(inputs[i-60: i, 0])

x_test = np.array(x_test) # tranformed into np array
x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))

###############################################
# make prediction with already trained rnn regressor with adam optimizer

from  featureExtraction import regressor

predicted_stock_price  = regressor.predict(x_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price) # to get the data out of matrix form



# form the data frame out of predicted stock price

predicted_stock_price = pd.DataFrame(predicted_stock_price)


##########################################################
# graphical visualization between the real and predicted data

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt 

plt.plot(real_stock_price, color='red', label='Googles real stock price')
plt.plot(predicted_stock_price, color= 'blue', label = 'Predicted stock price')
plt.title('Googles stock price prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.savefig('./v.eps', format = 'eps')
