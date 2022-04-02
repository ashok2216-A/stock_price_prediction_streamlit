import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import datetime
from datetime import date
from datetime import datetime
import yfinance as yf
import streamlit as st
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xg
# from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense,LSTM

START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
rawdata = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(rawdata.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=rawdata['Date'], y=rawdata['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=rawdata['Date'], y=rawdata['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

data = rawdata[['Close']]
# st.write(data.head())
futuredays = 50
data['prediction'] = data[['Close']].shift(-futuredays)
# st.write(data)

# st.subheader('x')
x = np.array(data.drop(['prediction'], 1))[:-futuredays]
# st.write(x)

# st.subheader('y')
y =np.array(data['prediction'])[:-futuredays]
# st.write(y)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>LinearRegression<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

st.write('Predicted by Linear Regressor')

x_train,x_test, y_train,y_test = train_test_split(x,y, test_size=0.25)
linear = LinearRegression().fit(x_train, y_train)

x_future = data.drop(['prediction'], 1)[:-futuredays]
x_future = x_future.tail(futuredays)
x_future = np.array(x_future)
st.subheader('future data')
# st.write(x_future)

linear_prediction = linear.predict(x_future)
# st.subheader('predicted data')
# st.write(linear_prediction)

inp_dt = pd.date_range(datetime.now(), periods=futuredays)
date = []
for dt in inp_dt:
    out = '{}'.format(dt.date())
    date.append(out)
# st.write(date)


def plot_pred_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=rawdata['Date'], y=rawdata['Open'], name="Stock Open"))
	fig.add_trace(go.Scatter(x=rawdata['Date'], y=rawdata['Close'], name="Stock Close"))
	fig.add_trace(go.Scatter(x =rawdata['Date'],y=data['Close'],name="Actual Data"))
	fig.add_trace(go.Scatter(x =date,y=linear_prediction,name="Predicted Data"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_pred_data()


# linear_prediction_train = linear.predict(x_train)

# st.write(x_train)
# def plot_pred1_data():
# 	fig = plt.figure()
# 	plt.plot(x_train)
# 	plt.plot(linear_prediction_train)
# 	st.plotly_chart(fig)
	
# plot_pred1_data()

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Decision_Tree<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
st.write('Predicted by Decision Tree Regressor')

x_train,x_test, y_train,y_test = train_test_split(x,y, test_size=0.25)
tree = DecisionTreeRegressor().fit(x_train, y_train)
tree_prediction = tree.predict(x_future)
# st.subheader('predicted data')

def plot_tree_pred_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=rawdata['Date'], y=rawdata['Open'], name="Stock Open"))
	fig.add_trace(go.Scatter(x=rawdata['Date'], y=rawdata['Close'], name="Stock Close"))
	fig.add_trace(go.Scatter(x =rawdata['Date'],y=data['Close'],name="Actual Data"))
	fig.add_trace(go.Scatter(x =date,y=tree_prediction,name="Predicted Data"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_tree_pred_data()

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Random_Forest<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
st.write('Predicted by Random Forest Regressor')

randforest_regr = RandomForestRegressor(max_depth=4, random_state=0)
randforest_regr.fit(x_train, y_train)
randforest_prediction = randforest_regr.predict(x_future)

def plot_rand_pred_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=rawdata['Date'], y=rawdata['Open'], name="Stock Open"))
	fig.add_trace(go.Scatter(x=rawdata['Date'], y=rawdata['Close'], name="Stock Close"))
	fig.add_trace(go.Scatter(x =rawdata['Date'],y=data['Close'],name="Actual Data"))
	fig.add_trace(go.Scatter(x =date,y=randforest_prediction,name="Predicted Data"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_rand_pred_data()

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>XG_BOOST<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
st.write('Predicted by XG Boost Regressor')
# Instantiation
xgb_r = xg.XGBRegressor(objective ='reg:linear',
                  n_estimators = 10, seed = 123)
  
# Fitting the model
xgb_r.fit(x_train, y_train)
  
# Predict the model
xg_boost_prediction = xgb_r.predict(x_future)

def plot_xgboost_pred_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=rawdata['Date'], y=rawdata['Open'], name="Stock Open"))
	fig.add_trace(go.Scatter(x=rawdata['Date'], y=rawdata['Close'], name="Stock Close"))
	fig.add_trace(go.Scatter(x =rawdata['Date'],y=data['Close'],name="Actual Data"))
	fig.add_trace(go.Scatter(x =date,y=xg_boost_prediction,name="Predicted Data"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_xgboost_pred_data()


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>LSTM<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# build LSTM model
# model=Sequential()
# model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
# model.add(LSTM(50,return_sequences=False))
# model.add(Dense(25))
# model.add(Dense(1))

# #copile the model
# model.compile(optimizer='adam',loss='mae',metrics='accuracy')

# #train the model
# model.fit(x_train,y_train,batch_size=10,epochs=5)

# lstm_prediction=model.predict(x_future)

# def plot_tree_pred_data():
# 	fig = go.Figure()
# 	fig.add_trace(go.Scatter(x=rawdata['Date'], y=rawdata['Open'], name="Stock Open"))
# 	fig.add_trace(go.Scatter(x=rawdata['Date'], y=rawdata['Close'], name="Stock Close"))
# 	fig.add_trace(go.Scatter(x =rawdata['Date'],y=data['Close'],name="Actual Data"))
# 	fig.add_trace(go.Scatter(x =date,y=lstm_prediction,name="Predicted Data"))
# 	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
# 	st.plotly_chart(fig)
	
# plot_tree_pred_data()
