from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import pandas as pd
import numpy as np

#local imports
from data import get_data

#this function is a personalized encoder
def encode_data(df):
    #encode cet values
    cet_encode = {'CORSO': 1, 'HAMICI': 2}
    for item in cet_encode.items():
        df['cet'] = df['cet'].replace([item[0]], item[1])
    
    #encode seasons values
    seasons = {"winter": 1, "spring":2, "summer":3, "autumn":4}
    for season in seasons.items():
        df['season'] = df['season'].replace([season[0]], season[1])

    #transform dates to ordinal
    df['date'] = df['date'].apply(lambda x: x.toordinal())

    le = LabelEncoder()
    df["code_town"] = le.fit_transform(df["code_town"])
    df["holiday"] = le.fit_transform(df["holiday"])
    return df


#this function prepares data for regression algorithms
def preprocess_data(df: pd.DataFrame):
    #remove useless columns
    df.pop("latitude")
    df.pop("longitude")
    df.pop("code_ticket")
    df.pop("year")
    df.pop("date_hijri")

    #change columns order and put the net column as class column
    df["class"] = df['net']
    df.pop("net")
    #encode data
    df = encode_data(df)
    #group data
    df['class'] = df.groupby(['date', 'code_town', 'cet'])['class'].transform('sum')
    df = df.drop_duplicates(subset=['date', 'code_town', 'cet'])
    print(df)
    #normalize data
    normalized = StandardScaler().fit_transform(df)

    return normalized

def gbrt():
    #get all data stored in db
    df = get_data()
    #clean, encode and normalize data
    prepared_data = preprocess_data(df)

    y = prepared_data[:, 6]
    X = prepared_data[:, :6]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    reg = GradientBoostingRegressor(random_state=0, n_estimators=5000)
    reg.fit(X_train, y_train)
    print(reg.score(X_test, y_test))
    y_pred = reg.predict(X_test)
    print(y_pred)
    print(y_test)

def svr():
    #get all data stored in db
    df = get_data()
    #clean, encode and normalize data
    prepared_data = preprocess_data(df)

    y = prepared_data[:, 6]
    X = prepared_data[:, :6]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    reg = SVR(C=5)
    reg.fit(X_train, y_train)
    print(reg.score(X_test, y_test))
    y_pred = reg.predict(X_test)
    print(y_pred)
    print(y_test)

def mlpr():
    #get all data stored in db
    df = get_data()
    #clean, encode and normalize data
    prepared_data = preprocess_data(df)

    y = prepared_data[:, 6]
    X = prepared_data[:, :6]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    reg = MLPRegressor(hidden_layer_sizes=(100,), activation="logistic", max_iter=2000)
    reg.fit(X_train, y_train)
    print(reg.score(X_test, y_test))
    y_pred = reg.predict(X_test)
    print(y_pred)
    print(y_test)


def main():
    mlpr()


if __name__ == "__main__":
    main()