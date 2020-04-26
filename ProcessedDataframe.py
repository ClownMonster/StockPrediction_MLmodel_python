'''
Return the data frame in the required Format for further analysis and modeling

'''

import pandas as pd 


def trainData():
    df_train = pd.read_csv('./Datasets/Google_Stock_Price_Train.csv', parse_dates=True)

    # since close and volume have object DataType they need to be converted to float or int
    df_train['Close'] = df_train['Close'].str.replace(',', '').astype(float)
    df_train['Volume'] = df_train['Volume'].str.replace(',', '').astype(float)
    return df_train

def testData():
    df_test = pd.read_csv('./Datasets/Google_Stock_Price_Test.csv')
    return df_test
