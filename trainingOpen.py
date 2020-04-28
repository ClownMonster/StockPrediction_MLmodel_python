'''
 Data preprosessing and reshaping
 Training only the data of open column from df

'''
# get the data and reduce to open set
from ProcessedDataframe import trainData
df = trainData()
open_trainig_set = df['Open']

import warnings
warnings.filterwarnings('ignore')

#################################################################

import numpy as np 

# converting the data set to np array  and reshaping
open_trainig_set = np.array(open_trainig_set)
open_trainig_set = open_trainig_set.reshape(-1,1)

###################################################################

# scaling the dataset
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
# transfrom every data to a particular value either 0 or < 1
training_set_scaled = sc.fit_transform(open_trainig_set)

###################################################################

x_train = []
y_train = []

# creating data structure with 60 timesteps with 1 output
for i in range(60,1258):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i,0])


x_train, y_train = np.array(x_train), np.array(y_train)

# reshaping
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
