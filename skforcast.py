import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from sklearn.metrics import mean_pinball_loss, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from joblib import dump, load
from data import add_hijri_holidays, get_data
from pattern_recognition import data_trend
import math, datetime

#this function is a personalized encoder
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


#this function prepares data for regression algorithms
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

    #change columns order and put the net column as class column
    df["y"] = df['net']/1000
    df['y'] = df['y'].astype(int)
    #group data
    #df['y'] = df.groupby(['date', 'code_town'])['y'].transform('sum')
    #df = df.drop_duplicates(subset=['date', 'code_town'])
    df = df.set_index('date')
    
    df = df.fillna(df.median())

    df.drop(['latitude', 'longitude', 'code_ticket', 'code_town', 'code_unity', 'net', 'date_hijri', 'cet', 'pop2016', 'pop2017', 'pop2018', 'pop2019', 'pop2020', 'pop2021','pop2022'], axis=1, inplace=True)    
    scaler = MinMaxScaler()
    df[['year', 'month', 'day', 'season', 'holiday', 'population', 'temperature', 'vent', 'y']] = scaler.fit_transform(df[['year', 'month', 'day', 'season', 'holiday', 'population', 'temperature', 'vent', 'y']])
    return df

def preprocess_data_weekly(df: pd.DataFrame):
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
    df['week'] = df.apply(lambda row: row['date'] - datetime.timedelta(days=row['date'].weekday()), axis=1)
    df['net'] = df.groupby('week')['net'].transform('sum')
    df.drop_duplicates('week', inplace=True)
    
    df2['date'] = df2.index
    df2['week'] = df2.apply(lambda row: row['date'] - datetime.timedelta(days=row['date'].weekday()), axis=1)
    df2['net'] = df2.groupby('week')['net'].transform('sum')
    df2.drop_duplicates('week', inplace=True)
    
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

    #change columns order and put the net column as class column
    df["y"] = df['net']/1000
    df['y'] = df['y'].astype(int)
    #group data
    #df['y'] = df.groupby(['date', 'code_town'])['y'].transform('sum')
    #df = df.drop_duplicates(subset=['date', 'code_town'])
    df = df.set_index('date')
    #df = df.asfreq('W')
    #df.fillna(0, inplace=True)
    df['y'].dropna(axis=0, inplace=True)
    df.drop(['latitude', 'longitude', 'week', 'code_ticket', 'code_town', 'code_unity', 'net', 'date_hijri', 'cet', 'pop2016', 'pop2017', 'pop2018', 'pop2019', 'pop2020', 'pop2021','pop2022'], axis=1, inplace=True)    
    return df

def prepare_data():
    df = get_data()
    data = preprocess_data(df)
    return data

def forcast(data):
    #new_data = predict_forcaster()
    print(data)
    #data = pd.concat([data, new_data], axis=0)
    data['y'] = data['y'].rolling(3).mean()
    data.dropna(axis=0, inplace=True)
    #print(data)
    data = data.reset_index(drop=True)
    steps = 30
    data_train = data.iloc[:-steps, :]
    data_test = data.iloc[-steps:, :]

    forecaster = ForecasterAutoreg(
                    regressor = CatBoostRegressor(random_state=0),
                    lags = 365
                )

    forecaster.fit(
        y = data_train['y'],
        exog = data_train[['year', 'month', 'day', 'season', 'holiday', 'population', 'temperature', 'vent']]
    )

    #dump(forecaster, filename='models/forcaster.py')
    predictions = forecaster.predict(
                steps = steps,
                exog = data_test[['year', 'month', 'day', 'season', 'holiday', 'population', 'temperature', 'vent']]
               )
    # Add datetime index to predictions
    predictions = pd.Series(data=predictions, index=data.index)
    print(predictions)
    fig, ax=plt.subplots(figsize=(9, 4))
    data_train['y'].plot(ax=ax, label='train')
    data_test['y'].plot(ax=ax, label='test')
    predictions.plot(ax=ax, label='predictions')
    ax.legend()
    plt.show()
    error_mse = mean_squared_error(
                y_true = data_test['y'].iloc[-steps:],
                y_pred = predictions.iloc[-steps:]
            )
    print(f"Test error (rmse): {math.sqrt(error_mse)}")
    y_true = data_test['y'].iloc[-steps:]
    y_pred = predictions.iloc[-steps:]
    print(f"Score: {r2_score(y_true, y_pred.to_numpy())}")

def interval_prediction(data):
    end_train = '2020-06-01 00:00:00'
    end_validation = '2021-06-01 00:00:00'
    data_train = data.loc[: end_train, :].copy()
    data_val   = data.loc[end_train:end_validation, :].copy()
    data_test  = data.loc[end_validation:, :].copy()

    print(f"Train dates      : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})")
    print(f"Validation dates : {data_val.index.min()} --- {data_val.index.max()}  (n={len(data_val)})")
    print(f"Test dates       : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")

    fig, ax=plt.subplots(figsize=(11, 4))
    data_train['y'].plot(label='train', ax=ax)
    data_val['y'].plot(label='validation', ax=ax)
    data_test['y'].plot(label='test', ax=ax)
    ax.legend()
    plt.show()

    forecaster = ForecasterAutoreg(
                regressor = XGBRegressor(),
                lags = 360
            )
    param_grid = {
        'eta': [0.9]
    }

    # Lags used as predictors
    lags_grid = [360]

    results_grid_q10 = grid_search_forecaster(
                                forecaster         = forecaster,
                                y                  = data.loc[:end_validation, 'y'],
                                param_grid         = param_grid,
                                lags_grid          = lags_grid,
                                steps              = 7,
                                refit              = True,
                                metric             = 'mean_squared_error',
                                initial_train_size = int(len(data_train)),
                                return_best        = True,
                                verbose            = False
                        )

    metric, predictions = backtesting_forecaster(
                            forecaster = forecaster,
                            y          = data['y'],
                            initial_train_size = len(data_train) + len(data_val),
                            steps      = 7,
                            refit      = True,
                            interval   = [10, 90],
                            n_boot     = 1000,
                            metric     = 'mean_squared_error',
                            verbose    = False
                        )
    print(predictions.head(4))
    inside_interval = np.where(
                     (data.loc[predictions.index, 'y'] >= predictions['lower_bound']) & \
                     (data.loc[predictions.index, 'y'] <= predictions['upper_bound']),
                     True,
                     False
                  )

    coverage = inside_interval.mean()
    print(f"Coverage of the predicted interval on test data: {100 * coverage}")

    fig, ax=plt.subplots(figsize=(11, 3))
    data.loc[end_validation:, 'y'].plot(ax=ax, label='y')
    ax.fill_between(
        predictions.index,
        predictions['lower_bound'],
        predictions['upper_bound'],
        color = 'deepskyblue',
        alpha = 0.3,
        label = '80% interval'
    )
    ax.legend()
    plt.show()


def predict_forcaster():
    #preparing new data to predict waste qte
    old_df = get_data()
    old_df["net"] = old_df.groupby(['date'])["net"].transform('sum')
    old_df.drop_duplicates(subset=['date'], inplace=True)
    old_df["date"] = pd.to_datetime(old_df["date"])
    old_df.sort_values(by='date', ascending=True, inplace=True)
    new_df = old_df[old_df["date"].dt.year == 2021]
    new_df['date'] = new_df['date'].apply(lambda x: x.replace(year = x.year + 1))
    add_hijri_holidays(new_df)
    new_df = preprocess_data(new_df)
    new_df['y'] = 0
    new_df = new_df[['year', 'month', 'day', 'season', 'holiday', 'population', 'y']]
    return new_df

def main():
    
    data = prepare_data()
    #interval_prediction(data)
    forcast(data)
    #predict_forcaster()



if __name__ == "__main__":
    main()