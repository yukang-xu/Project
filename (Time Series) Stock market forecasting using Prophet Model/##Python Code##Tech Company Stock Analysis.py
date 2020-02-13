#!/usr/bin/env python
# coding: utf-8

# In[1]:


###############################
# Identify the data
##############################
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # visualization library
import matplotlib.pyplot as plt # visualization library
from plotly.offline import init_notebook_mode, iplot # plotly offline mode
init_notebook_mode(connected=True) 
import plotly.graph_objs as go # plotly graphical object
import os
# import warnings library
import warnings        
# ignore filters
warnings.filterwarnings("ignore") 
plt.style.use('ggplot')
# Data Import
filepath1 = r"C:\Users\xuyuk\OneDrive - Georgia State University\Data import\Amazon.csv"
amazon = pd.read_csv(filepath1)
filepath2 = r"C:\Users\xuyuk\OneDrive - Georgia State University\Data import\Apple.csv"
apple = pd.read_csv(filepath2)
filepath3 = r"C:\Users\xuyuk\OneDrive - Georgia State University\Data import\Neflix.csv"
neflix = pd.read_csv(filepath3)
filepath4 = r"C:\Users\xuyuk\OneDrive - Georgia State University\Data import\Facebook.csv"
facebook = pd.read_csv(filepath4)
filepath5 = r"C:\Users\xuyuk\OneDrive - Georgia State University\Data import\Google.csv"
google = pd.read_csv(filepath5)


# In[2]:


# Amazon stock prediction
amazon["Date"] = pd.to_datetime(amazon["Date"])
plt.figure(figsize=(22,10))
plt.plot(amazon.Date,amazon.Close)
plt.title("amazon stock price")
plt.xlabel("Date")
plt.ylabel("price")
plt.show()
timeSeries = amazon.loc[:, ["Date","Close"]]
timeSeries.index = timeSeries.Date
ts = timeSeries.drop("Date",axis=1)
# using Prophet
from fbprophet import Prophet
#prophet reqiures a pandas df at the below config 
# ( date column named as DS and the value column as Y)
timeSeries.columns=['ds','y']
model = Prophet( yearly_seasonality=True) #instantiate Prophet with only yearly seasonality as our data is monthly 
model.fit(timeSeries)
# predict for 12 months in the furure and MS - month start is the frequency
future = model.make_future_dataframe(periods = 10, freq = 'MS')  
# now lets make the forecasts
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
model.plot(forecast)
forecast


# In[3]:


# Apple stock prediction
apple["Date"] = pd.to_datetime(apple["Date"])
plt.figure(figsize=(22,10))
plt.plot(apple.Date,apple.Close)
plt.title("apple stock price")
plt.xlabel("Date")
plt.ylabel("price")
plt.show()
timeSeries = apple.loc[:, ["Date","Close"]]
timeSeries.index = timeSeries.Date
ts = timeSeries.drop("Date",axis=1)
# using Prophet
from fbprophet import Prophet
#prophet reqiures a pandas df at the below config 
# ( date column named as DS and the value column as Y)
timeSeries.columns=['ds','y']
model = Prophet( yearly_seasonality=True) #instantiate Prophet with only yearly seasonality as our data is monthly 
model.fit(timeSeries)
# predict for 12 months in the furure and MS - month start is the frequency
future = model.make_future_dataframe(periods = 10, freq = 'MS')  
# now lets make the forecasts
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
model.plot(forecast)
forecast


# In[4]:


# Neflix stock prediction
neflix["Date"] = pd.to_datetime(neflix["Date"])
plt.figure(figsize=(22,10))
plt.plot(neflix.Date,neflix.Close)
plt.title("neflix stock price")
plt.xlabel("Date")
plt.ylabel("price")
plt.show()
timeSeries = neflix.loc[:, ["Date","Close"]]
timeSeries.index = timeSeries.Date
ts = timeSeries.drop("Date",axis=1)
# using Prophet
from fbprophet import Prophet
#prophet reqiures a pandas df at the below config 
# ( date column named as DS and the value column as Y)
timeSeries.columns=['ds','y']
model = Prophet( yearly_seasonality=True) #instantiate Prophet with only yearly seasonality as our data is monthly 
model.fit(timeSeries)
# predict for 12 months in the furure and MS - month start is the frequency
future = model.make_future_dataframe(periods = 10, freq = 'MS')  
# now lets make the forecasts
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
model.plot(forecast)
forecast


# In[5]:


# Facebook stock prediction
facebook["Date"] = pd.to_datetime(facebook["Date"])
plt.figure(figsize=(22,10))
plt.plot(facebook.Date,facebook.Close)
plt.title("apple stock price")
plt.xlabel("Date")
plt.ylabel("price")
plt.show()
timeSeries = facebook.loc[:, ["Date","Close"]]
timeSeries.index = timeSeries.Date
ts = timeSeries.drop("Date",axis=1)
# using Prophet
from fbprophet import Prophet
#prophet reqiures a pandas df at the below config 
# ( date column named as DS and the value column as Y)
timeSeries.columns=['ds','y']
model = Prophet( yearly_seasonality=True) #instantiate Prophet with only yearly seasonality as our data is monthly 
model.fit(timeSeries)
# predict for 12 months in the furure and MS - month start is the frequency
future = model.make_future_dataframe(periods = 10, freq = 'MS')  
# now lets make the forecasts
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
model.plot(forecast)
forecast


# In[6]:


# Google stock prediction
google["Date"] = pd.to_datetime(google["Date"])
plt.figure(figsize=(22,10))
plt.plot(google.Date,google.Close)
plt.title("google stock price")
plt.xlabel("Date")
plt.ylabel("price")
plt.show()
timeSeries = google.loc[:, ["Date","Close"]]
timeSeries.index = timeSeries.Date
ts = timeSeries.drop("Date",axis=1)
# using Prophet
from fbprophet import Prophet
#prophet reqiures a pandas df at the below config 
# ( date column named as DS and the value column as Y)
timeSeries.columns=['ds','y']
model = Prophet( yearly_seasonality=True) #instantiate Prophet with only yearly seasonality as our data is monthly 
model.fit(timeSeries)
# predict for 12 months in the furure and MS - month start is the frequency
future = model.make_future_dataframe(periods = 10, freq = 'MS')  
# now lets make the forecasts
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
model.plot(forecast)
forecast


# In[7]:


# stationary test for Amazon
# lets create time series from stock price
timeSeries = amazon.loc[:, ["Date","Close"]]
timeSeries.index = timeSeries.Date
ts = timeSeries.drop("Date",axis=1)

print("####################Rolling statistics######################")
#Rolling statistics# check stationary: mean, variance(std)
def check_mean_std(ts):
    rolmean = ts.rolling(6).mean()
    rolstd = ts.rolling(6).std()
    plt.figure(figsize=(22,10))   
    orig = plt.plot(ts, color='red',label='Original')
    mean = plt.plot(rolmean, color='black', label='Rolling Mean')
    std = plt.plot(rolstd, color='green', label = 'Rolling Std')
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.title('Rolling Mean & Standard Deviation')
    plt.legend()
    plt.show()
check_mean_std(ts)

print("####################Dickey-Fuller test######################")
# Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller
def check_adfuller(ts):
    result = adfuller(ts, autolag='AIC')
    print('Test statistic: ' , result[0])
    print('p-value: '  ,result[1])
    print('Critical Values:' ,result[4])
check_adfuller(ts.Close)

print("####################KPSS test######################")
# KPSS test
from statsmodels.tsa.stattools import kpss
kpsstest = kpss(ts.Close, regression='c')
kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
print(kpss_output)


# In[10]:


# dealing with non-stationary: differencing 
tsshift=ts.shift()
tsshift.iat[0,0]=ts.iat[521,0]
ts_diff = ts - tsshift
ts_diff = ts_diff.iloc[1:]
plt.figure(figsize=(22,10))
plt.plot(ts_diff)
plt.title("Differencing method") 
plt.xlabel("Date")
plt.ylabel("Differencing Price")
plt.show()
print("####################Rolling statistics######################")
check_mean_std(ts_diff)
print("####################Dickey-Fuller test######################")
check_adfuller(ts_diff.Close)
print("####################KPSS test######################")
kpsstest = kpss(ts_diff.Close, regression='c')
kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
print(kpss_output)


# In[16]:


# stationary test for Apple
# lets create time series from stock price
timeSeries = apple.loc[:, ["Date","Close"]]
timeSeries.index = timeSeries.Date
ts = timeSeries.drop("Date",axis=1)

print("####################Rolling statistics######################")
#Rolling statistics# check stationary: mean, variance(std)
def check_mean_std(ts):
    rolmean = ts.rolling(6).mean()
    rolstd = ts.rolling(6).std()
    plt.figure(figsize=(22,10))   
    orig = plt.plot(ts, color='red',label='Original')
    mean = plt.plot(rolmean, color='black', label='Rolling Mean')
    std = plt.plot(rolstd, color='green', label = 'Rolling Std')
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.title('Rolling Mean & Standard Deviation')
    plt.legend()
    plt.show()
check_mean_std(ts)

print("####################Dickey-Fuller test######################")
# Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller
def check_adfuller(ts):
    result = adfuller(ts, autolag='AIC')
    print('Test statistic: ' , result[0])
    print('p-value: '  ,result[1])
    print('Critical Values:' ,result[4])
check_adfuller(ts.Close)

print("####################KPSS test######################")
# KPSS test
from statsmodels.tsa.stattools import kpss
kpsstest = kpss(ts.Close, regression='c')
kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
print(kpss_output)

# dealing with non-stationary: differencing 
tsshift=ts.shift()
ts_diff = ts - tsshift
ts_diff = ts_diff.iloc[1:]
plt.figure(figsize=(22,10))
plt.plot(ts_diff)
plt.title("Differencing method") 
plt.xlabel("Date")
plt.ylabel("Differencing Price")
plt.show()
print("####################Rolling statistics######################")
check_mean_std(ts_diff)
print("####################Dickey-Fuller test######################")
check_adfuller(ts_diff.Close)
print("####################KPSS test######################")
kpsstest = kpss(ts_diff.Close, regression='c')
kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
print(kpss_output)


# In[12]:


# stationary test for Netflix
# lets create time series from stock price
timeSeries = neflix.loc[:, ["Date","Close"]]
timeSeries.index = timeSeries.Date
ts = timeSeries.drop("Date",axis=1)

print("####################Rolling statistics######################")
#Rolling statistics# check stationary: mean, variance(std)
def check_mean_std(ts):
    rolmean = ts.rolling(6).mean()
    rolstd = ts.rolling(6).std()
    plt.figure(figsize=(22,10))   
    orig = plt.plot(ts, color='red',label='Original')
    mean = plt.plot(rolmean, color='black', label='Rolling Mean')
    std = plt.plot(rolstd, color='green', label = 'Rolling Std')
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.title('Rolling Mean & Standard Deviation')
    plt.legend()
    plt.show()
check_mean_std(ts)

print("####################Dickey-Fuller test######################")
# Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller
def check_adfuller(ts):
    result = adfuller(ts, autolag='AIC')
    print('Test statistic: ' , result[0])
    print('p-value: '  ,result[1])
    print('Critical Values:' ,result[4])
check_adfuller(ts.Close)

print("####################KPSS test######################")
# KPSS test
from statsmodels.tsa.stattools import kpss
kpsstest = kpss(ts.Close, regression='c')
kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
print(kpss_output)

# dealing with non-stationary: differencing 
tsshift=ts.shift()
ts_diff = ts - tsshift
ts_diff = ts_diff.iloc[1:]
plt.figure(figsize=(22,10))
plt.plot(ts_diff)
plt.title("Differencing method") 
plt.xlabel("Date")
plt.ylabel("Differencing Price")
plt.show()
print("####################Rolling statistics######################")
check_mean_std(ts_diff)
print("####################Dickey-Fuller test######################")
check_adfuller(ts_diff.Close)
print("####################KPSS test######################")
kpsstest = kpss(ts_diff.Close, regression='c')
kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
print(kpss_output)


# In[15]:


# stationary test for Facebook
# lets create time series from stock price
timeSeries = facebook.loc[:, ["Date","Close"]]
timeSeries.index = timeSeries.Date
ts = timeSeries.drop("Date",axis=1)

print("####################Rolling statistics######################")
#Rolling statistics# check stationary: mean, variance(std)
def check_mean_std(ts):
    rolmean = ts.rolling(6).mean()
    rolstd = ts.rolling(6).std()
    plt.figure(figsize=(22,10))   
    orig = plt.plot(ts, color='red',label='Original')
    mean = plt.plot(rolmean, color='black', label='Rolling Mean')
    std = plt.plot(rolstd, color='green', label = 'Rolling Std')
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.title('Rolling Mean & Standard Deviation')
    plt.legend()
    plt.show()
check_mean_std(ts)

print("####################Dickey-Fuller test######################")
# Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller
def check_adfuller(ts):
    result = adfuller(ts, autolag='AIC')
    print('Test statistic: ' , result[0])
    print('p-value: '  ,result[1])
    print('Critical Values:' ,result[4])
check_adfuller(ts.Close)

print("####################KPSS test######################")
# KPSS test
from statsmodels.tsa.stattools import kpss
kpsstest = kpss(ts.Close, regression='c')
kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
print(kpss_output)

# dealing with non-stationary: differencing 
tsshift=ts.shift()
ts_diff = ts - tsshift
ts_diff = ts_diff.iloc[1:]
plt.figure(figsize=(22,10))
plt.plot(ts_diff)
plt.title("Differencing method") 
plt.xlabel("Date")
plt.ylabel("Differencing Price")
plt.show()
print("####################Rolling statistics######################")
check_mean_std(ts_diff)
print("####################Dickey-Fuller test######################")
check_adfuller(ts_diff.Close)
print("####################KPSS test######################")
kpsstest = kpss(ts_diff.Close, regression='c')
kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
print(kpss_output)


# In[18]:


# stationary test for Google
# lets create time series from stock price
timeSeries = google.loc[:, ["Date","Close"]]
timeSeries.index = timeSeries.Date
ts = timeSeries.drop("Date",axis=1)

print("####################Rolling statistics######################")
#Rolling statistics# check stationary: mean, variance(std)
def check_mean_std(ts):
    rolmean = ts.rolling(6).mean()
    rolstd = ts.rolling(6).std()
    plt.figure(figsize=(22,10))   
    orig = plt.plot(ts, color='red',label='Original')
    mean = plt.plot(rolmean, color='black', label='Rolling Mean')
    std = plt.plot(rolstd, color='green', label = 'Rolling Std')
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.title('Rolling Mean & Standard Deviation')
    plt.legend()
    plt.show()
check_mean_std(ts)

print("####################Dickey-Fuller test######################")
# Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller
def check_adfuller(ts):
    result = adfuller(ts, autolag='AIC')
    print('Test statistic: ' , result[0])
    print('p-value: '  ,result[1])
    print('Critical Values:' ,result[4])
check_adfuller(ts.Close)

print("####################KPSS test######################")
# KPSS test
from statsmodels.tsa.stattools import kpss
kpsstest = kpss(ts.Close, regression='c')
kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
print(kpss_output)

# dealing with non-stationary: differencing 
tsshift=ts.shift()
ts_diff = ts - tsshift
ts_diff = ts_diff.iloc[1:]
plt.figure(figsize=(22,10))
plt.plot(ts_diff)
plt.title("Differencing method") 
plt.xlabel("Date")
plt.ylabel("Differencing Price")
plt.show()
print("####################Rolling statistics######################")
check_mean_std(ts_diff)
print("####################Dickey-Fuller test######################")
check_adfuller(ts_diff.Close)
print("####################KPSS test######################")
kpsstest = kpss(ts_diff.Close, regression='c')
kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
print(kpss_output)


# In[ ]:




