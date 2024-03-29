
import pandas as pd
import numpy as np
from math import sqrt
from datetime import datetime
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, GRU
from keras.optimizers import *
from keras.preprocessing.sequence import TimeseriesGenerator
import math
from data import add_hijri_holidays, get_data
from catboost import CatBoostRegressor
from skforcast import preprocess_data_weekly

from utilities import openPkl, savePkl


def encode_data(df):
    #encode cet values
    #cet_encode = {'CORSO': 1, 'HAMICI': 2}
    #for item in cet_encode.items():
    #    df['cet'] = df['cet'].replace([item[0]], item[1])
    
    #encode seasons values
    seasons = {"winter": 0.1, "spring":0.2, "summer":0.3, "autumn":0.4}
    for season in seasons.items():
        df['season'] = df['season'].replace([season[0]], season[1])
    #transform dates
    df['date'] = pd.to_datetime(df['date']) 
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    #encode holidays
    holidays = {"normal": 0.1, "Ramadhan":0.2, "Eid al-Fitr":0.3, "Eid al-Adha":0.4, "Islamic New Year": 0.5, "Ashura": 0.6, "Prophet's Birthday": 0.7}
    for holiday in holidays.items():
        df['holiday'] = df['holiday'].replace([holiday[0]], holiday[1])
    
    #encode towns
    towns = np.unique(df['code_town'].to_numpy())
    for town in towns:
        code = int(town[1:])
        df['code_town'] = df['code_town'].replace([town], code)
    
    return df



def preprocess_data(df: pd.DataFrame):
    df = df[df["code_town"] != 'S001']
    df['date'] = pd.to_datetime(df['date'])    
    df['net'] = df.groupby(by=['date', 'code_town'])['net'].transform('sum')
    df.drop_duplicates(subset=['code_town', 'date'], inplace=True)
    df2 = df.groupby(by='date').agg({
        'net': 'sum', 
        'pop2016': 'sum', 
        'pop2017': 'sum', 
        'pop2018': 'sum',
        'pop2019': 'sum',
        'pop2020': 'sum',
        'pop2021': 'sum',
        'pop2022': 'sum'
    })
    df['net'] = df.groupby('date')['net'].transform('sum')
    df.drop_duplicates('date', inplace=True)
    
    #sort data by date
    df.sort_values(by='date', ascending=True, inplace=True)
    
    #insert population column
    df['population'] = 1
    df.loc[df['date'].dt.year == 2016, 'population'] = df[df2.index.year == 2016]['pop2016']
    df.loc[df['date'].dt.year == 2017, 'population'] = df[df2.index.year == 2017]['pop2017']
    df.loc[df['date'].dt.year == 2018, 'population'] = df[df2.index.year == 2018]['pop2018']
    df.loc[df['date'].dt.year == 2019, 'population'] = df[df2.index.year == 2019]['pop2019']
    df.loc[df['date'].dt.year == 2020, 'population'] = df[df2.index.year == 2020]['pop2020']
    df.loc[df['date'].dt.year == 2021, 'population'] = df[df2.index.year == 2021]['pop2021']
    df.loc[df['date'].dt.year == 2022, 'population'] = df[df2.index.year == 2022]['pop2022']

    #encode data
    df = encode_data(df)
    df = df.set_index('date')
    
    df['weekday'] = [i.isoweekday() for i in df.index] 
    df['is_weekday'] = df.apply(lambda row: row['weekday'] == 5 or row['weekday']==6, axis=1)
    #change columns order and put the net column as class column
    df["y"] = df['net']/1000
    df['y'] = df['y'].astype(int)
    #df['y'] = df['y'].rolling(7).mean()
    df.dropna(axis=0, inplace=True)
    #group data
    #df['y'] = df.groupby(['date', 'code_town'])['y'].transform('sum')
    #df = df.drop_duplicates(subset=['date', 'code_town'])
    

    df.drop(['latitude', 'longitude', 'pressure', 'weekday', 'code_ticket', 'code_town', 'code_unity', 'net', 'date_hijri', 'cet', 'pop2016', 'pop2017', 'pop2018', 'pop2019', 'pop2020', 'pop2021','pop2022'], axis=1, inplace=True)    
    return df

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = pd.DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = pd.concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df



# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return pd.Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = np.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]


def main():
    df = get_data()
    raw_values = preprocess_data(df)['y'].values

    data = timeseries_to_supervised(raw_values).values
    print(data)
    train, test = data[0: -365], data[-365:]
    
    scaler, train_scaled, test_scaled = scale(train, test)

    # fit the model
    lstm_model = fit_lstm(train_scaled, 1, 3000, 4)
    # forecast the entire training dataset to build up state for forecasting
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    lstm_model.predict(train_reshaped, batch_size=1)

    # walk-forward validation on the test data
    predictions = list()
    for i in range(len(test_scaled)):
        # make one-step forecast
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = forecast_lstm(lstm_model, 1, X)
        # invert scaling
        yhat = invert_scale(scaler, X, yhat)
        # invert differencing
        yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
        # store forecast
        predictions.append(yhat)
        expected = raw_values[len(train) + i + 1]
        print('day=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))
        # report performance
        rmse = sqrt(mean_squared_error(raw_values[-365:], predictions))
        print('Test RMSE: %.3f' % rmse)
        # line plot of observed vs predicted
        pyplot.plot(raw_values[-365:])
        pyplot.plot(predictions)
        pyplot.show()


## convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        #print("dagi")
        #for j in range(i, i+look_back):
        #    print(dataset[j,8])

        a = dataset[i:(i+look_back), 8]
        exog = []
        exog.extend(dataset[i+look_back])
        for s in a:
            exog.append(s)
        y = exog.pop(8)
        dataX.append(exog)
        dataY.append(y)
    return np.array(dataX), np.array(dataY)

#def create_dataset(dataset, look_back=1):
#	dataX, dataY = [], []
#	for i in range(len(dataset)-look_back-1):
#		a = dataset[i:(i+look_back), 0]
#		dataX.append(a)
#		dataY.append(dataset[i + look_back, 0])
#	return np.array(dataX), np.array(dataY)

# fix random seed for reproducibility
np.random.seed(7)
# load the dataset
df = get_data()
dataframe = preprocess_data(get_data())
df = dataframe[['day', 'month', 'year', 'temperature', 'vent', 'population', 'holiday', 'season', 'y']]
dataset = df.to_numpy()
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.75)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
#print(trainY)
dataX, dataY = create_dataset(dataset, look_back) 
# reshape input to be [samples, time steps, features]
#trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
#testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
#print(trainX)
#print(trainY)
# create and fit the LSTM network
batch_size = 1
#model = Sequential()
###LSTM model
#model.add(LSTM(50, batch_input_shape=(batch_size, look_back+8, 1), stateful=True))
#model.add(Dense(1))
#model.compile(loss='mean_squared_error', optimizer='adam')
#print('model created')
#loss = []
#val_loss = []
#for i in range(20):
#    history = model.fit(dataX, dataY, epochs=1, validation_split=0.2, batch_size=batch_size, verbose=2, shuffle=False)
#    loss.append(history.history['loss'])
#    val_loss.append(history.history['val_loss'])
#    model.reset_states()
#
#pyplot.plot(loss)
#pyplot.plot(val_loss)
#pyplot.title('model loss')
#pyplot.ylabel('loss')
#pyplot.xlabel('epoch')
#pyplot.legend(['train', 'val'], loc='upper left')
#pyplot.show()

#ANN model
#model.add(GRU(50, input_shape=(look_back+8, 1)))
#model.add(Dense(1))
#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#model.fit(trainX, trainY, epochs=20, batch_size=batch_size, shuffle=False)
#print('finished training')

#MLP model
#model.add(Dense(20, activation='relu', input_dim=dataX.shape[1]))
#model.add(Dense(10))
#model.add(Dense(1))
#model.compile(loss='mean_squared_error', optimizer='adam')
#model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2, shuffle=False)

# make predictions
#savePkl(model, 'models/gru365.pkl')
#model = openPkl('models/gru.pkl')
#trainPredict = model.predict(trainX, batch_size=batch_size)
#model.reset_states()
#testPredict = model.predict(testX, batch_size=batch_size)


# invert predictions
#trainPredict = scaler.inverse_transform(trainPredict)
#trainY = scaler.inverse_transform([trainY])
#testPredict = scaler.inverse_transform(testPredict)
#testY = scaler.inverse_transform([testY])
##calculate root mean squared error
#trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
#print('Train Score: %.4f RMSE' % (trainScore))
#testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
#print('Test Score: %.4f RMSE' % (testScore))
#r2score = r2_score(testY, testPredict[:,0])
#print('test Score%.4f R2' % (r2score))
## shift train predictions for plotting
#trainPredictPlot = np.empty_like(dataset)
#trainPredictPlot[:, :] = np.nan
#trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
## shift test predictions for plotting
#testPredictPlot = np.empty_like(dataset)
#testPredictPlot[:, :] = np.nan
#testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
## plot baseline and predictions
##pyplot.plot(dataset)
#pyplot.plot(trainPredictPlot)
#pyplot.plot(testPredictPlot)
#pyplot.show()

df = get_data()
df['date'] = pd.to_datetime(df['date'])
new_df = df[df['date'].dt.year >= 2021]


#dataframe = preprocess_data(get_data())[['day', 'month', 'year', 'temperature', 'vent', 'population', 'holiday', 'season', 'y']]
#old_df = dataframe[dataframe.index >= datetime(2020, 12, 30)]
#print(old_df)
#new_df = old_df[old_df.index.year == 2021].copy()
#new_df['date'] = new_df.index
new_df['date'] = new_df['date'].apply(lambda x: x.replace(year = x.year + 1))
new_df = add_hijri_holidays(new_df)
new_df = preprocess_data(new_df)[['day', 'month', 'year', 'temperature', 'vent', 'population', 'holiday', 'season', 'y']]


print(new_df)
new_df = new_df.iloc[10:]
print(new_df)
model = openPkl('models/lstmexog.pkl')

scale_array = new_df['y'].to_numpy()
scaler2 = MinMaxScaler()
test = scaler2.fit_transform(scale_array.reshape(-1, 1))


new_dataset = new_df.to_numpy()
new_dataset = new_dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
new_dataset = scaler.fit_transform(new_dataset)
new_datasetX, new_datasetY = create_dataset(new_dataset, 3)
new_datasetX = np.reshape(new_datasetX, (new_datasetX.shape[0], new_datasetX.shape[1], 1))
y_pred = model.predict(new_datasetX, batch_size=batch_size)
predicted = scaler2.inverse_transform(y_pred)
#savePkl(predicted, "predictions/prediction2022.pkl")

#pyplot.plot(new_datasetY, label='Observed')
#pyplot.plot(predicted, label='Prediction 2022')
#pyplot.legend(loc="upper left")
#pyplot.show()
prediction_df = pd.DataFrame(predicted, columns = ['net'])
start = pd.to_datetime('2022-01-01')
rng = pd.date_range(start, periods=350)
df = pd.DataFrame({'date': rng, 'net': prediction_df['net']})
#pyplot.plot(df['net'])
#pyplot.show()

df.to_pickle("predictions/predictions_df_2022.pkl")
#print(prediction_df)
#new_df = new_df.set_index(new_df['date'])
#new_df.drop('date', axis=1, inplace=True)
#new_df['y'] = new_df['y'].apply(lambda x: 0)
#
#print(new_df)
#
#scale_array = old_df['y'].to_numpy()
#scaler2 = MinMaxScaler()
#test = scaler2.fit_transform(scale_array.reshape(-1, 1))
##
#model = openPkl('models/lstmexog.pkl')
#i = 1
#for index, row in new_df.iterrows():
#    old_df = old_df.append(row, ignore_index=True)
#    print(old_df)
#    new_dataset = old_df.to_numpy()
#    new_dataset = new_dataset.astype('float32')
#    # normalize the dataset
#    scaler = MinMaxScaler(feature_range=(0, 1))
#    new_dataset = scaler.fit_transform(new_dataset)
#    print(new_dataset)
#    print("dagi")
#    new_datasetX, new_datasetY = create_dataset(new_dataset, 365)
#    new_datasetX = np.reshape(new_datasetX, (new_datasetX.shape[0], new_datasetX.shape[1], 1))
#    size = new_datasetX.shape[0]
#    print("dataset created")
#    last_row = new_datasetX[size-1]
#    print(last_row)
#    print(dataX)
#    y_pred = model.predict(last_row, batch_size=batch_size)
#    model.reset_states()
#    predicted = scaler2.inverse_transform(y_pred)
#    #print(predicted)
#    old_df.iloc[-1, old_df.columns.get_loc('y')] = predicted[0]
#    
#print(old_df)
predicted_year = old_df[old_df.index > 364]['y'].to_numpy()
print(predicted_year)
old_data = old_df[old_df.index < 365]['y'].to_numpy()
print(old_data)
pyplot.plot(old_data)
pyplot.plot(predicted)
pyplot.show()
            
    
    