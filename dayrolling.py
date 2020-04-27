'''
Prints the data rolling back of  7days from the day need to visualize

'''
from ProcessedDataframe import trainData

def mean_data():
    train_df = trainData()
    d = train_df.rolling(7).mean()
    return d
