from typing import Optional

from fastapi import FastAPI
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from database.database import connect

import petl as etl
import pandas as pd
from datetime import datetime, date
import matplotlib.pyplot as plt
import numpy as np
from workalendar.africa import Algeria
from hijri_converter import Hijri
from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import normalize, LabelEncoder, KBinsDiscretizer
from meteostat import Daily, Point, Stations
from routers import map, statistics
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#include routers to main app
app.include_router(map.router)
app.include_router(statistics.router)

@app.get("/db")
def get_data():
    db_connection = connect()

    rotation_table = etl.fromdb(db_connection, 'select r.*, c.longitude, c.latitude from rotation r, commune c where c.code = r.code_town')
    ticket_table = etl.fromdb(db_connection, 'select * from ticket')
    print(rotation_table)

    #start = datetime(2016, 1, 1)
    #end = datetime(2019, 12, 31)
#
    ## Get daily data
    #city = Point(36.7206251, 3.1854975)
    #data = Daily(city, start, end)
    #data = data.fetch()
#
    #print(data)

    ticket_table = etl.rename(ticket_table, {'code': 'code_ticket'})
    data = etl.merge(rotation_table, ticket_table, key=['cet', 'date', 'code_ticket'])
    data = etl.cutout(data, 'brute', 'montant', 'id', 'code_ticket')
    print(data)
    etl.topickle(data, "data_cordinates.pkl")
    #return etl.todataframe(data)


def encode_data(df):
    #encode cet values
    cet_encode = {'CORSO': 1, 'HAMICI': 2}
    for item in cet_encode.items():
        df['cet'] = df['cet'].replace([item[0]], item[1])
    
    #transform dates to ordinal
    df['date'] = df['date'].apply(lambda x: x.toordinal())
    df['date_hijri'] = df['date_hijri'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').toordinal())

    le = LabelEncoder()
    df["id_vehicle"] = le.fit_transform(df["id_vehicle"])
    df["code_town"] = le.fit_transform(df["code_town"])

    return df

def insert_holidays(val, row):
    #get list of islamic holidays
    cal = Algeria()
    islamic_holidays = cal.get_islamic_holidays()

    #hijri date is in yyyy-mm-dd format
    #split hijri date
    hijri_date = str(row["date_hijri"]).split("-")
    #check month is Ramadhan
    if hijri_date[1] == '09':
        return "Ramadhan"
    else: #check for holidays
        for tuple in islamic_holidays:
            #index 0 is month, index 1 is day, index 2 is label
            if int(hijri_date[1]) == tuple[0] and int(hijri_date[2]) == tuple[1]:
                return tuple[2]

    return "normal"

def get_hijri_month(val, row):
    hijri_date = str(row["date_hijri"]).split("-")
    return hijri_date[1]

def get_temperature(date, lat, long):
    start = datetime(2016, 1, 1)
    end = datetime.today()
    # Get daily data
    city = Point(lat, long)
    data = Daily(city, start, end)
    data = data.fetch()
    
    return data.loc[date]['tavg']





@app.get("/")
def read_root():
    print("get data")
    df = etl.todataframe(etl.frompickle("data.pkl"))
    df.pop("heure")
    
    print("encode data")
    df = encode_data(df)
    df.pop("id_vehicle")
    df['net'] = df.groupby(['date', 'date_hijri', 'code_town', 'cet'])['net'].transform('sum')
    df = df.drop_duplicates(subset=['date', 'date_hijri', 'code_town', 'cet'])
    print(df)
    print("normalize data")
    normalized_df = normalize(df)
    print("ok")
    
    y = normalized_df[:,4]
    X = normalized_df[:,:4]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    reg = GradientBoostingRegressor(random_state=0, n_estimators=500)
    reg.fit(X_train, y_train)
    print(reg.score(X_test, y_test))
    y_pred = reg.predict(X_test)
    print(y_pred)
    print(y_test)

    regr = MLPRegressor(random_state=1,hidden_layer_sizes=(400,) ,max_iter=1000000)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    print(y_pred)
    print(y_test)
    print(regr.score(X_test, y_test))

    return {"Hello": "World"}



@app.get("/test")
def test():
    date = datetime.date(1431, 6, 15)
    print(date.toordinal())

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}