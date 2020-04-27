import pandas as pd


import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt 

import plotly.express as px


from  ProcessedDataframe import trainData , testData

# reading the train data into df_train
df_train = trainData()

###############################################################################################

'''
 >> print(df_train.isna().any())  
  checks whether the data has any None entry which cannot be used
  i,e non applicable values

'''

#############################################################################################

# prints the basic information reguarding the data frame
def basicInfo_ofData(dataframe):
    print(dataframe.info())
    print(dataframe.isna().any())


##############################################################################################

# Visualization of each type against date


def OpenDataVisualization():
    global df_train
    fig = px.bar(df_train,x='Date',y='Open', color='Open')
    fig.show()



def CloseDataVisualization():
    global df_train
    fig = px.bar(df_train,x='Date',y='Close', color='Close')
    fig.show()



def HighDataVisualization():
    global df_train
    fig = px.bar(df_train,x='Date',y='High', color='High')
    fig.show()
    return



def LowDataVisualization():
    global df_train
    fig = px.bar(df_train,x='Date',y='Low', color='Low')
    fig.show()


def VolumeDV():
    global df_train
    fig = px.bar(df_train,x='Date',y='Volume', color='Volume')
    fig.show()


#############################################################################################

# visualizing 7day rolling mean data


from dayrolling import mean_data

def roll_Visualization():
    mean_df = mean_data()

    # fig = px.scatter(mean_df, y = 'Open',).update_traces(mode = 'lines+markers')
    # fig = px.scatter(mean_df, y = 'Close',).update_traces(mode = 'lines+markers')
    fig = px.scatter(mean_df, y = 'Volume').update_traces(mode = 'lines+markers')
    # fig = px.scatter(mean_df, y = 'Low').update_traces(mode = 'lines+markers')
    # fig = px.scatter(mean_df, y = 'High').update_traces(mode = 'lines+markers')
    '''
    uncomment any one of the above line to get respective visualization
    '''

    fig.show()

    
 #######################################################################################