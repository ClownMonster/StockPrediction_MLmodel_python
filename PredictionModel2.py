'''
Predicting the future stock price using Prophet

'''

from fbprophet import Prophet
import pandas as pd

df = pd.read_csv('./Datasets/Google_Stock_Price_Train.csv')
open_data = df.groupby('Date').sum()['Open'].reset_index()
open_data.columns = ['ds', 'y'] # renaming the columns as required by model

open_data['ds'] = pd.to_datetime(open_data['ds']) # formating date to required form to feed to the model



m = Prophet(interval_width = 0.96)
m.fit(open_data)

future_predict = m.make_future_dataframe(periods = 50)

my_forecast = m.predict(future_predict)

predicted_data = my_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


#############################################################################
# graphical visualization
import matplotlib
matplotlib.use('Agg')

fig = m.plot(predicted_data)

fig.savefig('./v2.eps', format = 'eps')

