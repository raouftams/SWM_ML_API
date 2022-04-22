from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pattern_recognition import data_trend
from utilities import savePkl, openPkl

#local imports
from data import add_hijri_holidays, get_data

#this function is a personalized encoder
def encode_data(df):
    #encode cet values
    #cet_encode = {'CORSO': 1, 'HAMICI': 2}
    #for item in cet_encode.items():
    #    df['cet'] = df['cet'].replace([item[0]], item[1])
    
    #encode seasons values
    seasons = {"winter": 1, "spring":2, "summer":3, "autumn":4}
    for season in seasons.items():
        df['season'] = df['season'].replace([season[0]], season[1])

    #transform dates
    df['date'] = pd.to_datetime(df['date']) 
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    #encode holidays
    holidays = {"normal": 1, "Ramadhan":2, "Eid al-Fitr":3, "Eid al-Adha":4, "Islamic New Year": 5, "Ashura": 6, "Prophet's Birthday": 7}
    for holiday in holidays.items():
        df['holiday'] = df['holiday'].replace([holiday[0]], holiday[1])
    
    #encode towns
    towns = np.unique(df['code_town'].to_numpy())
    for town in towns:
        code = int(town[1:])
        print(town, code)
        df['code_town'] = df['code_town'].replace([town], code)
    
    return df


#this function prepares data for regression algorithms
def preprocess_data(df: pd.DataFrame):
    df["net"] = df.groupby(['code_town', 'date'])["net"].transform('sum')
    df.drop_duplicates(subset=['code_town', 'date'], inplace=True)
    df = df[df["code_town"] != 'S001']
    #sort data by date
    df.sort_values(by='date', ascending=True, inplace=True)

    #get data trend
    trend = data_trend(df.copy(), 'net', model='additive', period=365).to_frame('values')
    df['date'] = pd.to_datetime(df['date'])
    df = df.merge(trend, on='date', how='inner')
    df['values'] = df['values'].astype(int)
    #insert population column
    df['population'] = 1
    df.loc[df['date'].dt.year == 2016, 'population'] = df[df['date'].dt.year == 2016]['pop2016']
    df.loc[df['date'].dt.year == 2017, 'population'] = df[df['date'].dt.year == 2017]['pop2017']
    df.loc[df['date'].dt.year == 2018, 'population'] = df[df['date'].dt.year == 2018]['pop2018']
    df.loc[df['date'].dt.year == 2019, 'population'] = df[df['date'].dt.year == 2019]['pop2019']
    df.loc[df['date'].dt.year == 2020, 'population'] = df[df['date'].dt.year == 2020]['pop2020']
    df.loc[df['date'].dt.year == 2021, 'population'] = df[df['date'].dt.year == 2021]['pop2021']
    #encode data
    df = encode_data(df)
    #shift t-1 and t+1 of class data
    for i in range(1, 10):
        col = 't-'+str(i)
        df[col] = df['values'].shift(i)

    df.dropna(axis=0, inplace=True)
    #change columns order and put the net column as class column
    df["class"] = df['values']
    #group data
    df['class'] = df.groupby(['date', 'code_town'])['class'].transform('sum')
    df = df.drop_duplicates(subset=['date', 'code_town'])

    df.drop(['latitude', 'longitude', 'code_ticket', 'code_unity', 'holiday', 'season', 'date', 'net', 'values', 'date_hijri', 'cet', 'pop2016', 'pop2017', 'pop2018', 'pop2019', 'pop2020', 'pop2021','pop2022'], axis=1, inplace=True)
    
    #normalize data
    #normalized = StandardScaler().fit_transform(df)
    #return normalized

    return df.to_numpy()

def gbrt():
    #get all data stored in db
    df = get_data()
    df = df[df["code_town"] == 'C002']
    #clean, encode and normalize data
    prepared_data = preprocess_data(df)
    
    y = prepared_data[:, 14]
    X = prepared_data[:, :14]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False, random_state=0)
    
    reg = GradientBoostingRegressor(learning_rate=0.1, n_estimators=1000)
    reg.fit(X_train, y_train)
    print(reg.score(X_test, y_test))
    y_pred = reg.predict(X_test)
    print(y_pred[:10])
    print(y_test[:10])
    plt.plot(y_test)
    plt.plot(y_pred)
    plt.show()
    
    #savePkl(reg, "models/gbrt_by_town.pkl")

def svr():
    #get all data stored in db
    df = get_data()
    df = df[df["code_town"] == 'C002']
    #clean, encode and normalize data
    prepared_data = preprocess_data(df)

    y = prepared_data[:, 4]
    X = prepared_data[:, :4]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    reg = SVR(C=1)
    reg.fit(X_train, y_train)
    print(reg.score(X_test, y_test))
    y_pred = reg.predict(X_test)
    print(y_pred[:5])
    print(y_test[:5])

def mlpr():
    #get all data stored in db
    df = get_data()
    df = df[df["code_town"] == 'C002']
    #clean, encode and normalize data
    prepared_data = preprocess_data(df)

    y = prepared_data[:, 14]
    X = prepared_data[:, :14]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False, random_state=0)

    reg = MLPRegressor(hidden_layer_sizes=(160,), activation="relu", solver='adam', max_iter=200)
    reg.fit(X_train, y_train)
    print(reg.score(X_test, y_test))
    y_pred = reg.predict(X_test)
    print(y_pred[:5])
    print(y_test[:5])
    plt.plot(y_test)
    plt.plot(y_pred)
    plt.show()


def main():
    mlpr()
    gbrt()
    """
    df = get_data()
    df["net"] = df.groupby(['code_town', 'date'])["net"].transform('sum')
    df.drop_duplicates(subset=['code_town', 'date'], inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df['code_town'] == 'C002']
    df = df[df["date"].dt.year == 2021]
    print(df)
    df['date'] = df['date'].apply(lambda x: x.replace(year = x.year + 1))
    df.sort_values(by='date', ascending=True, inplace=True)
    df.drop_duplicates(subset=['date', 'code_town'], inplace=True)
    dates = df['date'].to_numpy()
    plt.plot(df["date"], df["net"])
    add_hijri_holidays(df)
    array = preprocess_data(df)
    X = array[:, :4]
    model = openPkl("models/gbrt_by_town.pkl")
    y_pred = model.predict(X)
    trend_df = pd.DataFrame({ 'date': np.array(dates), 'net': np.array(y_pred)})
    trend_df.drop_duplicates('date', inplace=True)
    print(trend_df)
    trend = data_trend(trend_df.copy(), 'net', model='additive', period=1).to_frame('values')
    plt.plot(trend.index, trend['values'])
    plt.show()
    """
if __name__ == "__main__":
    main()