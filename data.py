import pandas as pd
import numpy as np
from datetime import datetime, date
from meteostat import Daily, Point
import swifter
import matplotlib.pyplot as plt
from workalendar.africa import Algeria
from sklearn.preprocessing import LabelEncoder

#local imports
from database.database import connect
from utilities import savePkl, openPkl


#initialize a pycopg2 database connection
DB_CONNECTION = connect()

#this function reads rotation table from db and returns a df
def get_town_data():
    sql = 'select code as code_town, latitude, longitude, pop2016, pop2017, pop2018, pop2019, pop2020, pop2021, pop2022, code_unity from commune'
    return pd.read_sql_query(sql, DB_CONNECTION)

#this function reads ticket table from db and returns a df
def get_ticket_data():
    sql = 'select code as code_ticket, date, cet, net from ticket'
    return pd.read_sql_query(sql, DB_CONNECTION)

#this function reads town table from db and returns df
def get_rotation_data():
    sql = 'select date, date_hijri, cet, code_ticket, code_town from rotation'
    return pd.read_sql_query(sql, DB_CONNECTION)

#this function gets and stores weather data for each town and date
def get_weather_data():
    #initializing the start and end date of historical weather data
    start = datetime(2016, 1, 1)
    end = datetime.today()

    #initialize a dict for weather data df of each town
    weather_data = {}

    #get towns data
    town_df = get_town_data()
    #make sure indexes pair with number of rows
    town_df = town_df.reset_index()  
    #iterate over rows
    for index, row in town_df.iterrows():
        # Get daily data
        city = Point(row['latitude'], row['longitude'])
        data = Daily(city, start, end)
        data = data.fetch()
        weather_data[row['code_town']] = data

    #save data to pickle file
    savePkl(weather_data, 'data/weather_data.pkl')


#this function reads and merges data from tables (rotation, ticket, town)
def read_and_merge_tables():
    #read tables
    rotation_df = get_rotation_data()
    ticket_df = get_ticket_data()
    town_df = get_town_data()
    #merge all tables
    total_df = rotation_df.merge(ticket_df, how='inner', on=['date', 'cet', 'code_ticket'])
    total_df = total_df.merge(town_df, how='inner', on='code_town')
    #group data 
    total_df['net'] = total_df.groupby(['date', 'date_hijri', 'code_town', 'cet'])['net'].transform('sum')
    total_df = total_df.drop_duplicates(subset=['date', 'date_hijri', 'code_town', 'cet'])
    
    return total_df

#this function return air temperature in specific place (latitude and longitude) at specific date
def get_temperature(date, town):
    #transform date to str format 'yyyy-mm-dd'
    date_str = date.strftime("%Y-%m-%d")
    #read weather data from pickle file
    data_dict = openPkl('data/weather_data.pkl')
    try:
        return data_dict[town].loc[date_str]['tavg'] 
    except:
        return np.nan()

#this function return air temperature in specific place (latitude and longitude) at specific date
def get_wind_speed(date, town):
    #transform date to str format 'yyyy-mm-dd'
    date_str = date.strftime("%Y-%m-%d")
    #read weather data from pickle file
    data_dict = openPkl('data/weather_data.pkl')
    try:
        return data_dict[town].loc[date_str]['wspd'] 
    except:
        return np.nan()

def get_wind_dir(date, town):
    #transform date to str format 'yyyy-mm-dd'
    date_str = date.strftime("%Y-%m-%d")
    #read weather data from pickle file
    data_dict = openPkl('data/weather_data.pkl')
    try:
        return data_dict[town].loc[date_str]['wdir'] 
    except:
        return np.nan()


def get_pressure(date, town):
    #transform date to str format 'yyyy-mm-dd'
    date_str = date.strftime("%Y-%m-%d")
    #read weather data from pickle file
    data_dict = openPkl('data/weather_data.pkl')
    try:
        return data_dict[town].loc[date_str]['pres'] 
    except:
        return np.nan()

def get_daily_precipitation(date, town):
    #transform date to str format 'yyyy-mm-dd'
    date_str = date.strftime("%Y-%m-%d")
    #read weather data from pickle file
    data_dict = openPkl('data/weather_data.pkl')
    try:
        return data_dict[town].loc[date_str]['prcp'] 
    except:
        return np.nan()

#this function adds weather data to our dataframe
def add_weather_to_data(df):
    df['temperature'] = df.swifter.apply(lambda row: get_temperature(row['date'], row['code_town']), axis=1)
    df['vent'] = df.swifter.apply(lambda row: get_wind_speed(row['date'], row['code_town']), axis=1)
    df['wind_dir'] = df.swifter.apply(lambda row: get_wind_dir(row['date'], row['code_town']), axis=1)
    df['pressure'] = df.swifter.apply(lambda row: get_pressure(row['date'], row['code_town']), axis=1)
    df['rain'] = df.swifter.apply(lambda row: get_daily_precipitation(row['date'], row['code_town']), axis=1)
    
    return df

#this function gets a hijri data and returns an islamic holiday 
def get_holiday(hijri_date):
    #get list of islamic holidays
    cal = Algeria()
    islamic_holidays = cal.get_islamic_holidays()

    #hijri date is in yyyy-mm-dd format
    #split hijri date
    date = str(hijri_date).split("-")
    #check month is Ramadhan
    if date[1] == '09':
        return "Ramadhan"
    else: #check for holidays
        for tuple in islamic_holidays:
            #index 0 is month, index 1 is day, index 2 is label
            if int(date[1]) == tuple[0] and int(date[2]) == tuple[1]:
                return tuple[2]

    return "normal"

#this function adds hijri holidays to data
def add_hijri_holidays(df):
    df["holiday"] = df.apply(lambda row: get_holiday(row["date_hijri"]), axis=1)
    return df

#this function separates month from hijri date
def get_hijri_month(df):
    df["hijri_month"] = df.apply(lambda row: datetime.strptime(row["date_hijri"], '%Y-%m-%d').month, axis=1)
    return df

#this function separates year from date
def get_year(df):
    df['year'] = df.apply(lambda row: row['date'].year, axis=1)
    return df

#this function returns the season given a date
def get_season(current_date):
    y = 2000 # dummy leap year to allow input X-02-29 (leap day)
    seasons = [('winter', (date(y,  1,  1),  date(y,  3, 20))),
            ('spring', (date(y,  3, 21),  date(y,  6, 20))),
            ('summer', (date(y,  6, 21),  date(y,  9, 22))),
            ('autumn', (date(y,  9, 23),  date(y, 12, 20))),
            ('winter', (date(y, 12, 21),  date(y, 12, 31)))]
    if isinstance(current_date, datetime):
        current_date = current_date.date()
    current_date = current_date.replace(year=y)
    return next(season for season, (start, end) in seasons
                if start <= current_date <= end)

#this function add season to dataframe
def add_seasons(df):
    df["season"] = df.apply(lambda row: get_season(row["date"]), axis=1)
    return df

#this function gets day name from date and add "day" column to df
def add_days(df):
    df["day"] = df.apply(lambda row: row["date"].strftime("%A"), axis=1)
    return df

#get rotations number by town and date
def get_rotations_number():
    sql = 'SELECT r.code_town, c.code_unity, r.date, count(r.*) as rotations from rotation r, commune c where r.code_town = c.code group by(code_unity, code_town, date)'
    return pd.read_sql_query(sql, DB_CONNECTION)

def get_rotations_by_hour():
    sql = "select r.code_town, r.heure, c.code_unity, count(r.*) as rotations from rotation r, commune c where date >= '2021-01-01' and r.code_town = c.code group by(code_unity, code_town, heure)"
    return pd.read_sql_query(sql, DB_CONNECTION)

def get_efficiency_by_hour():
    sql = "select r.code_town, r.heure, c.code_unity, (t.net/1000)/v.volume as compact_rate from rotation r, commune c, vehicle v, ticket t where r.date >= '2021-01-01' and r.code_town = c.code and v.code = r.id_vehicle and v.volume != 0 and t.code = r.code_ticket and t.cet = r.cet and t.date = r.date"
    return pd.read_sql_query(sql, DB_CONNECTION)

def get_transform_data():
    df = read_and_merge_tables()
    get_weather_data()    
    df = add_weather_to_data(df)
    df = add_hijri_holidays(df)
    df = add_seasons(df)
    df = get_year(df)
    df.to_pickle("data/final_data.pkl")
    

def get_data():
    return pd.read_pickle("data/final_data.pkl")


#get_transform_data()
#df = pd.read_pickle("data/final_data.pkl")
#df = add_days(df)
#print(df)
#df.to_pickle("data/final_data1.pkl")