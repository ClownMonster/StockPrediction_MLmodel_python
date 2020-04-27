'''
 data preprosessing
 training only the data of open col from df

'''
# get the data and reduce to open set
from ProcessedDataframe import trainData
df = trainData()
open_trainig_set = df['Open']

import numpy as np 
# scaling the dataset
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0,1))
# transfrom to a particular value either 0 or < 1
training_set_scaled = sc.fit_transform(open_trainig_set)

x_train = []
y_train = []

# creating data structure with 60 timesteps
for i in range(60,1258):
    #x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i,0])

#x_train, y_train = np.array(x_train), np.array(y_train)
print(y_train)
