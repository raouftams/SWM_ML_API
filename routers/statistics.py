from fastapi import APIRouter
import json

#local imports
from database.database import connect
from table.RotationTable import RotationTable
from table.TownTable import TownTable
from table.VehicleTable import VehicleTable
from table.TicketTable import TicketTable
from pattern_recognition import *
from utilities import NumpyArrayEncoder


db_connection = connect()

router = APIRouter()

"""---------------General information------------------"""

#get number of used and owned vehicles, number of rotations and total waste qte by town
@router.get(path="/stats/info/town/{id}")
def get_town_information(id):
    rotation_table = RotationTable()
    vehicle_table = VehicleTable()
    ticket_table = TicketTable()
    town_table = TownTable()

    data = {
        "used_vehicles": int(float(rotation_table.get_town_used_vehicles(id, db_connection)[0])),
        "rotations_by_day": int(float(rotation_table.get_town_nb_rotations_day(id, db_connection)[0])),
        "rotations": int(float(rotation_table.get_town_nb_rotations(id, db_connection)[0])),
        "waste_qte": int(float(ticket_table.get_total_waste_quantity(id, db_connection)[0])),
        "ratio" : float(town_table.get_ratio(id, db_connection)[0]),
        "population" : int(town_table.get_population(id, db_connection)[0]),
    }
    return json.dumps(data)

#get the number of rotations and total waste qte
@router.get(path="/stats/info/all-towns")
def get_all_towns_information():
    rotation_table = RotationTable()
    ticket_table = TicketTable()
    vehicle_table = VehicleTable()

    data = {
        "rotations": int(float(rotation_table.get_nb_rotations(db_connection)[0])),
        "waste_qte": int(float(ticket_table.get_total_waste(db_connection)[0])),
        "vehicles": vehicle_table.get_nb_vehicles(db_connection)[0]
    }
    return json.dumps(data)

#get number of used vehicles, number of all rotations, average of rotations by day and total waste qte by unity
@router.get(path='/stats/info/unity/{code}')
def get_unity_information(code):
    rotation_table = RotationTable()
    ticket_table = TicketTable()
    town_table = TownTable()

    data = {
        "used_vehicles": int(float(rotation_table.get_unity_used_vehicles(code, db_connection)[0])),
        "rotations_by_day": int(float(rotation_table.get_unity_nb_rotations_day(code, db_connection)[0])),
        "rotations": int(float(rotation_table.get_unity_nb_rotations(code, db_connection)[0])),
        "waste_qte": int(float(ticket_table.get_unity_total_waste_quantity(code, db_connection)[0])),
        "ratio": float(town_table.get_unity_ratio(code, db_connection)[0]),
        "population": int(town_table.get_unity_population(code, db_connection)[0])
    }
    return json.dumps(data)

"""---------------Waste Trend----------------------"""

#get waste generation trend by (day, month, year) for town
@router.get(path="/stats/trend/town/{town}/{period}")
def get_trend_by_town(town, period):
    data = {}
    if period == 'year':
        data = trend_by_town(365)
    if period == 'day':
        data = trend_by_town(1)
    if period == 'month':
        data = trend_by_town(30)

    #selecting the town
    town_data = data[town]
    json_data ={
        "labels": town_data.index.astype(str).to_numpy(),
        "values": town_data["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#get waste generation trend by period, town, and year
@router.get(path='/stats/trend/town/{town}/{period}/{year}')
def get_trend_by_town_year(town, period, year):
    if period == 'year':
        data = trend_by_town_year(town, 182, int(year))
    if period == 'day':
        data = trend_by_town_year(town, 1, int(year))
    if period == 'month':
        data = trend_by_town_year(town, 30, int(year))

    json_data ={
        "labels": data.index.astype(str).to_numpy(),
        "values": data["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#get waste generation trend for all towns
@router.get(path="/stats/all-towns/trend/{period}")
def get_trend(period):
    predictions = pd.read_pickle("predictions/predictions_df_2022.pkl")
    if period == 'year':
        data = all_towns_trend_prediction(365)
    if period == 'day':
        data = all_towns_trend_prediction(1)
    if period == 'month':
        data = all_towns_trend_prediction(30)

    #selecting the town
    #data.index = data.index.strftime('%d-%m-%y')
    json_data ={
        "labels": data.index.astype(str).to_numpy(),
        "values": data["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#get waste generation trend for all towns for one year
@router.get(path='/stats/all-towns/trend/{period}/{year}')
def get_trend_year(period, year):
    if period == 'year':
        data = all_towns_trend_year(182, int(year))
    if period == 'day':
        data = all_towns_trend_year(1, int(year))
    if period == 'month':
        data = all_towns_trend_year(30, int(year))

    json_data ={
        "labels": data.index.astype(str).to_numpy(),
        "values": data["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#get waste generation trend for all towns
@router.get(path="/stats/trend/unity/{code}/{period}")
def get_trend_by_unity(code, period):
    data = {}
    if period == 'year':
        data = trend_by_unity(365)
    if period == 'day':
        data = trend_by_unity(1)
    if period == 'month':
        data = trend_by_unity(30)

    #selecting the unity df
    df = data[code]
    json_data = {
        "labels": df.index.astype(str).to_numpy(),
        "values": df["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#get waste generation trend by period, town, and year
@router.get(path='/stats/trend/unity/{unity}/{period}/{year}')
def get_trend_by_unity_year(unity, period, year):
    if period == 'year':
        data = trend_by_unity_year(unity, 182, int(year))
    if period == 'day':
        data = trend_by_unity_year(unity, 1, int(year))
    if period == 'month':
        data = trend_by_unity_year(unity, 30, int(year))

    
    json_data ={
        "labels": data.index.astype(str).to_numpy(),
        "values": data["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

"""-----------------------Waste Seasonality---------------------------"""

#get waste generation seaonality for all towns
@router.get(path="/stats/all-towns/seasonality/{period}")
def get_seasonality(period):
    if period == 'year':
        data = all_towns_seasonality(365)
    if period == 'day':
        data = all_towns_seasonality(1)
    if period == 'month':
        data = all_towns_seasonality(30)

    json_data ={
        "labels": data.index.astype(str).to_numpy(),
        "values": data["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#get waste generation seasonality by (day, month, year) for town
@router.get(path="/stats/seasonality/town/{town}/{period}")
def get_seasonality_by_town(town, period):
    data = {}
    if period == 'year':
        data = seasonality_by_town(365)
    if period == 'day':
        data = seasonality_by_town(1)
    if period == 'month':
        data = seasonality_by_town(30)

    #selecting the town
    town_data = data[town]
    json_data ={
        "labels": town_data.index.astype(str).to_numpy(),
        "values": town_data["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#get waste generation seasonality by (day, month, year) for unity
@router.get(path="/stats/seasonality/unity/{code}/{period}")
def get_seasonality_by_unity(code, period):
    data = {}
    if period == 'year':
        data = seasonality_by_unity(365)
    if period == 'day':
        data = seasonality_by_unity(1)
    if period == 'month':
        data = seasonality_by_unity(30)

    #selecting the town
    town_data = data[code]
    json_data ={
        "labels": town_data.index.astype(str).to_numpy(),
        "values": town_data["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)


"""--------------------Rotations trend--------------------"""
#number of rotations trend for all towns
@router.get(path="/stats/rotations/all-towns/trend/{period}")
def get_rotations_trend(period):
    if period == 'year':
        data = rotations_trend(365)
    if period == 'day':
        data = rotations_trend(1)
    if period == 'month':
        data = rotations_trend(30)

    
    json_data ={
        "labels": data.index.astype(str).to_numpy(),
        "values": data["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#number of rotations trend for all towns by year
@router.get(path="/stats/rotations/all-towns/trend/{period}/{year}")
def get_rotations_trend_year(period, year):
    if period == 'year':
        data = rotations_trend_year(182, int(year))
    if period == 'day':
        data = rotations_trend_year(1, int(year))
    if period == 'month':
        data = rotations_trend_year(30, int(year))

    json_data ={
        "labels": data.index.astype(str).to_numpy(),
        "values": data["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#number of rotations trend by (day, month, year) for town
@router.get(path="/stats/rotations/trend/town/{town}/{period}")
def get_rotations_trend_by_town(town, period):
    data = {}
    if period == 'year':
        data = rotations_trend_by_town(365)
    if period == 'day':
        data = rotations_trend_by_town(1)
    if period == 'month':
        data = rotations_trend_by_town(30)

    #selecting the town
    town_data = data[town]
    json_data ={
        "labels": town_data.index.astype(str).to_numpy(),
        "values": town_data["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#number of rotations trend by (day, month, year) for town by year
@router.get(path="/stats/rotations/trend/town/{town}/{period}/{year}")
def get_rotations_trend_by_town_year(town, period, year):
    data = {}
    if period == 'year':
        data = rotations_trend_by_town_year(town, 182, int(year))
    if period == 'day':
        data = rotations_trend_by_town_year(town, 1, int(year))
    if period == 'month':
        data = rotations_trend_by_town_year(town, 30, int(year))

    json_data ={
        "labels": data.index.astype(str).to_numpy(),
        "values": data["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#number of rotations trend by (day, month, year) for a unity
@router.get(path="/stats/rotations/trend/unity/{code}/{period}")
def get_rotations_trend_by_unity(code, period):
    data = {}
    if period == 'year':
        data = rotations_trend_by_unity(365)
    if period == 'day':
        data = rotations_trend_by_unity(1)
    if period == 'month':
        data = rotations_trend_by_unity(30)

    #selecting the unity df
    df = data[code]
    json_data = {
        "labels": df.index.astype(str).to_numpy(),
        "values": df["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#number of rotations trend by (day, month, year) for unity by year
@router.get(path="/stats/rotations/trend/unity/{unity}/{period}/{year}")
def get_rotations_trend_by_unity_year(unity, period, year):
    data = {}
    if period == 'year':
        data = rotations_trend_by_unity_year(unity, 182, int(year))
    if period == 'day':
        data = rotations_trend_by_unity_year(unity, 1, int(year))
    if period == 'month':
        data = rotations_trend_by_unity_year(unity, 30, int(year))

    json_data ={
        "labels": data.index.astype(str).to_numpy(),
        "values": data["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#number of rotations trend by hour for town
@router.get(path="/stats/rotations/trend/all-towns/hour")
def get_rotations_trend_hour():
    data = rotations_trend_by_hour()
    #compact_rate_data = compact_rate_trend_hour()
    json_data ={
        "rotations":{
            "labels": data['heure'].astype(str).to_numpy(),
            "values": data["rotations"].to_numpy()   
        },
        #"compact_rate":{
        #    "labels": compact_rate_data['heure'].astype(str).to_numpy(),
        #    "values": compact_rate_data["compact_rate"].to_numpy()
        #}
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#compaction rate by hour for towns
@router.get(path="/stats/rotations/trend/all-towns/compact-rate/hour")
def get_compact_rate_hour():
    data = compact_rate_trend_hour()
    json_data ={
        "labels": data['heure'].astype(str).to_numpy(),
        "values": data["compact_rate"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#number of rotations trend by (day, month, year) for town
@router.get(path="/stats/rotations/trend/hour/town/{town}")
def get_rotations_trend_by_town_hour(town):
    data = rotations_trend_by_town_hour(town)
    #compact_rate_data = compact_rate_trend_town_hour(town)
    json_data ={
        "rotations":{
            "labels": data['heure'].astype(str).to_numpy(),
            "values": data["rotations"].to_numpy()   
        },
        #"compact_rate":{
        #    "labels": compact_rate_data['heure'].astype(str).to_numpy(),
        #    "values": compact_rate_data["compact_rate"].to_numpy()
        #}
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#number of rotations trend by (day, month, year) for unity
@router.get(path="/stats/rotations/trend/hour/unity/{unity}")
def get_rotations_trend_by_unity_hour(unity):
    data = rotations_trend_by_unity_hour(unity)
    compact_rate_data = compact_rate_trend_unity_hour(unity)
    json_data ={
        "rotations":{
            "labels": data['heure'].astype(str).to_numpy(),
            "values": data["rotations"].to_numpy()   
        },
        "compact_rate":{
            "labels": compact_rate_data['heure'].astype(str).to_numpy(),
            "values": compact_rate_data["compact_rate"].to_numpy()
        }
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

"""-----------------Holidays data-------------------------"""

#get waste quantity by hijri holidays
@router.get(path="/stats/waste/holidays/town/{town}")
def get_holiday_waste_by_town(town):
    data = holiday_qte_town(town)
    labels = data["holiday"].to_numpy()
    data.index = data["holiday"]
    data = data["net"]
    data_json = {
        "labels": labels,
        "values": data.to_dict()
    }

    return(json.dumps(data_json, cls=NumpyArrayEncoder))

#get waste quantity by holidays/unity
@router.get(path="/stats/waste/holidays/unity/{code}")
def get_holiday_waste_by_unity(code):
    data = holiday_qte_unity(code)
    labels = data["holiday"].to_numpy()
    data.index = data["holiday"]
    data = data["net"]
    data_json = {
        "labels": labels,
        "values": data.to_dict()
    }

    return(json.dumps(data_json, cls=NumpyArrayEncoder))

#get waste quantity by hijri holidays
@router.get(path="/stats/waste/holidays")
def get_holiday_waste_qte():
    data = holiday_qte()
    labels = data["holiday"].to_numpy()
    data.index = data["holiday"]
    data = data["net"]
    data_json = {
        "labels": labels,
        "values": data.to_dict()
    }

    return(json.dumps(data_json, cls=NumpyArrayEncoder))

"""-----------------Seasons data----------------------"""

#get waste quantity by season
@router.get(path="/stats/waste/season/town/{town}")
def get_season_waste_by_town(town):
    data = season_qte_town(town)
    labels = data["season"].to_numpy()
    data.index = data["season"]
    data = data["net"]
    data_json = {
        "labels": labels,
        "values": data.to_dict()
    }

    return(json.dumps(data_json, cls=NumpyArrayEncoder))

#get waste quantity by season/unity
@router.get(path="/stats/waste/season/unity/{code}")
def get_season_waste_by_unity(code):
    data = season_qte_unity(code)
    labels = data["season"].to_numpy()
    data.index = data["season"]
    data = data["net"]
    data_json = {
        "labels": labels,
        "values": data.to_dict()
    }

    return(json.dumps(data_json, cls=NumpyArrayEncoder))

#get waste quantity by season for all towns
@router.get(path="/stats/waste/season")
def get_season_waste_qte():
    data = season_qte()
    labels = data["season"].to_numpy()
    data.index = data["season"]
    data = data["net"]
    data_json = {
        "labels": labels,
        "values": data.to_dict()
    }

    return(json.dumps(data_json, cls=NumpyArrayEncoder))


"""------------------season's waste qte change rate------------------"""

#get qte waste change rate for all towns
@router.get(path="/stats/waste/change-rate-by-season")
def get_qte_change_rate_by_season():
    data = qte_change_rate_by_season()
    seasons = {'winter': -5, 'summer': 10, 'spring': 5, 'autumn': -1}
    summer = []
    winter = []
    spring = []
    autumn = []
    for season in seasons.keys():
        for town in data.keys():
            if data[town][season] >= seasons[season]:
                if season == 'summer':
                    summer.append(town)
                if season == 'winter':
                    winter.append(town)
                if season == 'spring':
                    spring.append(town)
                if season == 'autumn':
                    autumn.append(town)
    
    json_data = {
        'summer': summer,
        'winter': winter,
        'spring': spring,
        'autumn': autumn
    }

    return json.dumps(json_data)


"-------------------------Stats by time filter ------------------"
#get registred years and months
@router.get(path="/statistics/temporel/years-months")
def get_years_months():
    rotation_table = RotationTable()
    data = rotation_table.get_years_and_months(db_connection)
    data = data.astype(int)
    data.sort_values(['year', 'month'], ascending=True, inplace=True)
    dict_data =  data.groupby('year')['month'].apply(list).to_dict()
    return dict_data


#get stats for all towns by year and month
@router.get(path="/statistics/all-towns/info/{year}/{month}")
def get_all_towns_info_year_month(year, month):
    rotation_table = RotationTable()
    town_table = TownTable()
    population = town_table.get_population_year(year, db_connection)
    data = rotation_table.get_info_by_year_month_all_towns(year, month, db_connection)
    data.fillna(0, axis=0, inplace=True)
    json_data = {
        "nb_rotations_day": data['nb_rotations_day'].to_numpy()[0],
        "nb_rotations": data['nb_rotations'].to_numpy()[0],
        "nb_vehicles": data['nb_vehicles'].to_numpy()[0],
        "waste_qte": data['waste_qte'].to_numpy()[0],
        "waste_qte_day": data['waste_qte_day'].to_numpy()[0],
        "compact_rate": data['compact_rate'].to_numpy()[0],
        "ratio": data['waste_qte_day'].to_numpy()[0]*1000/float(population[0])
    }

    return json.dumps(json_data, cls=NumpyArrayEncoder)


#get efficiency by vehicle mark, year and month
@router.get(path="/statistics/all-towns/vehicle-efficiency-mark/{year}/{month}")
def get_vehicles_mark_efficiency_year_month(year, month):
    vehicle_table = VehicleTable()
    data = vehicle_table.get_efficiency_by_mark_year_month(year, month, db_connection)
    data['compact_rate'] = data.groupby('marque')['compact_rate'].transform('mean')
    data.drop_duplicates('marque', inplace=True)
    data = data[data['marque'] != 'nan']
    data.sort_values(by='compact_rate', ascending=False, inplace=True)
    data.dropna(axis=0, inplace=True)
    json_data = {
        "labels": data['marque'].to_numpy(),
        "values": data['compact_rate'].to_numpy()
    }

    return json.dumps(json_data, cls=NumpyArrayEncoder)


#get efficiency by vehicle volume, year and month
@router.get(path="/statistics/all-towns/vehicle-efficiency-volume/{year}/{month}")
def get_vehicles_volume_efficiency_year_month(year, month):
    vehicle_table = VehicleTable()
    data = vehicle_table.get_efficiency_by_volume_year_month(year, month, db_connection)
    data['compact_rate'] = data.groupby('volume')['compact_rate'].transform('mean')
    data.drop_duplicates('volume', inplace=True)
    data = data[data['volume'] != 0]
    data.sort_values(by='compact_rate', ascending=False, inplace=True)
    data.dropna(axis=0, inplace=True)
    json_data = {
        "labels": data['volume'].astype(str).to_numpy(),
        "values": data['compact_rate'].to_numpy()
    }

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#get hour waste trend by year and month
@router.get(path="/statistics/all-towns/waste-trend-hour/{year}/{month}")
def get_all_towns_hour_waste_trend_year_month(year, month):
    ticket_table = TicketTable()
    data = ticket_table.get_all_towns_waste_year_month(year, month, db_connection)
    data['heure'] = pd.to_datetime(data['heure'])
    data['heure'] = data['heure'].apply(lambda x: x.replace(minute=0, second=0))
    data['waste_qte'] = data.groupby(by='heure')['waste_qte'].transform('sum')
    data.drop_duplicates(subset='heure', inplace=True)
    data.sort_values(by='heure', ascending=True, inplace=True)
    data.dropna(axis=0, inplace=True)
    json_data = {
        "labels": data['heure'].astype(str).to_numpy(),
        "values": data['waste_qte'].to_numpy()
    }

    return json.dumps(json_data, cls=NumpyArrayEncoder)


"""------------------- stats by town and unity ------------------"""
#get stats for a given town by year and month
@router.get(path="/statistics/town/{code}/info/{year}/{month}")
def get_all_towns_info_year_month(code, year, month):
    rotation_table = RotationTable()
    town_table = TownTable()
    population = town_table.get_population_year_town(code, year, db_connection)
    data = rotation_table.get_info_by_year_month_town(code, year, month, db_connection)
    data.fillna(0, axis=0, inplace=True)
    json_data = {
        "nb_rotations_day": data['nb_rotations_day'].to_numpy()[0],
        "nb_rotations": data['nb_rotations'].to_numpy()[0],
        "nb_vehicles": data['nb_vehicles'].to_numpy()[0],
        "waste_qte": data['waste_qte'].to_numpy()[0],
        "waste_qte_day": data['waste_qte_day'].to_numpy()[0],
        "compact_rate": data['compact_rate'].to_numpy()[0],
        "ratio": data['waste_qte_day'].to_numpy()[0]*1000/float(population[0])
    }

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#get stats for a given unity by year and month
@router.get(path="/statistics/unity/{code}/info/{year}/{month}")
def get_all_towns_info_year_month(code, year, month):
    rotation_table = RotationTable()
    town_table = TownTable()
    population = town_table.get_population_year_unity(code, year, db_connection)
    data = rotation_table.get_info_by_year_month_unity(code, year, month, db_connection)
    data.fillna(0, axis=0, inplace=True)
    json_data = {
        "nb_rotations_day": data['nb_rotations_day'].to_numpy()[0],
        "nb_rotations": data['nb_rotations'].to_numpy()[0],
        "nb_vehicles": data['nb_vehicles'].to_numpy()[0],
        "waste_qte": data['waste_qte'].to_numpy()[0],
        "waste_qte_day": data['waste_qte_day'].to_numpy()[0],
        "compact_rate": data['compact_rate'].to_numpy()[0],
        "ratio": data['waste_qte_day'].to_numpy()[0]*1000/float(population[0])
    }

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#get efficiency by vehicle mark, year and month for given town
@router.get(path="/statistics/town/{code}/vehicle-efficiency-mark/{year}/{month}")
def get_vehicles_mark_efficiency_year_month_town(code, year, month):
    vehicle_table = VehicleTable()
    data = vehicle_table.get_efficiency_by_mark_year_month_town(code, year, month, db_connection)
    data['compact_rate'] = data.groupby('marque')['compact_rate'].transform('mean')
    data.drop_duplicates('marque', inplace=True)
    data = data[data['marque'] != 'nan']
    data.sort_values(by='compact_rate', ascending=False, inplace=True)
    data.dropna(axis=0, inplace=True)
    json_data = {
        "labels": data['marque'].to_numpy(),
        "values": data['compact_rate'].to_numpy()
    }

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#get efficiency by vehicle mark, year and month for given unity
@router.get(path="/statistics/unity/{code}/vehicle-efficiency-mark/{year}/{month}")
def get_vehicles_mark_efficiency_year_month_unity(code, year, month):
    vehicle_table = VehicleTable()
    data = vehicle_table.get_efficiency_by_mark_year_month_unity(code, year, month, db_connection)
    data['compact_rate'] = data.groupby('marque')['compact_rate'].transform('mean')
    data.drop_duplicates('marque', inplace=True)
    data = data[data['marque'] != 'nan']
    data.sort_values(by='compact_rate', ascending=False, inplace=True)
    data.dropna(axis=0, inplace=True)
    json_data = {
        "labels": data['marque'].to_numpy(),
        "values": data['compact_rate'].to_numpy()
    }

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#get efficiency by vehicle volume, year and month for a given town
@router.get(path="/statistics/town/{code}/vehicle-efficiency-volume/{year}/{month}")
def get_vehicles_volume_efficiency_year_month_town(code, year, month):
    vehicle_table = VehicleTable()
    data = vehicle_table.get_efficiency_by_volume_year_month_town(code, year, month, db_connection)
    data['compact_rate'] = data.groupby('volume')['compact_rate'].transform('mean')
    data.drop_duplicates('volume', inplace=True)
    data = data[data['volume'] != 0]
    data.sort_values(by='compact_rate', ascending=False, inplace=True)
    data.dropna(axis=0, inplace=True)
    json_data = {
        "labels": data['volume'].astype(str).to_numpy(),
        "values": data['compact_rate'].to_numpy()
    }

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#get efficiency by vehicle volume, year and month for a given unity
@router.get(path="/statistics/unity/{code}/vehicle-efficiency-volume/{year}/{month}")
def get_vehicles_volume_efficiency_year_month_unity(code, year, month):
    vehicle_table = VehicleTable()
    data = vehicle_table.get_efficiency_by_volume_year_month_unity(code, year, month, db_connection)
    data['compact_rate'] = data.groupby('volume')['compact_rate'].transform('mean')
    data.drop_duplicates('volume', inplace=True)
    data = data[data['volume'] != 0]
    data.sort_values(by='compact_rate', ascending=False, inplace=True)
    data.dropna(axis=0, inplace=True)
    json_data = {
        "labels": data['volume'].astype(str).to_numpy(),
        "values": data['compact_rate'].to_numpy()
    }

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#get hour waste trend by year and month for a given town
@router.get(path="/statistics/town/{code}/waste-trend-hour/{year}/{month}")
def get_all_towns_hour_waste_trend_year_month_town(code, year, month):
    ticket_table = TicketTable()
    data = ticket_table.get_all_towns_waste_year_month_town(code, year, month, db_connection)
    data['heure'] = pd.to_datetime(data['heure'])
    data['heure'] = data['heure'].apply(lambda x: x.replace(minute=0, second=0))
    data['waste_qte'] = data.groupby(by='heure')['waste_qte'].transform('sum')
    data.drop_duplicates(subset='heure', inplace=True)
    data.sort_values(by='heure', ascending=True, inplace=True)
    data.dropna(axis=0, inplace=True)
    json_data = {
        "labels": data['heure'].astype(str).to_numpy(),
        "values": data['waste_qte'].to_numpy()
    }

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#get hour waste trend by year and month for a given town
@router.get(path="/statistics/unity/{code}/waste-trend-hour/{year}/{month}")
def get_all_towns_hour_waste_trend_year_month_unity(code, year, month):
    ticket_table = TicketTable()
    data = ticket_table.get_all_towns_waste_year_month_unity(code, year, month, db_connection)
    data['heure'] = pd.to_datetime(data['heure'])
    data['heure'] = data['heure'].apply(lambda x: x.replace(minute=0, second=0))
    data['waste_qte'] = data.groupby(by='heure')['waste_qte'].transform('sum')
    data.drop_duplicates(subset='heure', inplace=True)
    data.sort_values(by='heure', ascending=True, inplace=True)
    data.dropna(axis=0, inplace=True)
    json_data = {
        "labels": data['heure'].astype(str).to_numpy(),
        "values": data['waste_qte'].to_numpy()
    }

    return json.dumps(json_data, cls=NumpyArrayEncoder)



#get waste quantity by day of week for given  year and month  
@router.get(path="/statistics/all-towns/waste-quantity-day/{year}/{month}")
def get_all_towns_day_waste_year_month_all_towns(year, month):
    ticket_table = TicketTable()
    data = ticket_table.get_all_towns_waste_year_month_days(year, month, db_connection)
    data['waste_qte'] = data.groupby(by='day')['waste_qte'].transform('mean')
    data.drop_duplicates(subset='day', inplace=True)
    data.sort_values(by='day', ascending=True, inplace=True)
    data.dropna(axis=0, inplace=True)
    days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    data['day'] = data['day'].astype(int)
    data['day'] = data['day'].apply(lambda x: days[x-1])
    json_data = {
        "labels": data['day'].astype(str).to_numpy(),
        "values": data['waste_qte'].to_numpy()
    }

    return json.dumps(json_data, cls=NumpyArrayEncoder)


#get waste quantity by day of week for given  year and month  and town
@router.get(path="/statistics/town/{code}/waste-quantity-day/{year}/{month}")
def get_day_waste_year_month_town(code, year, month):
    ticket_table = TicketTable()
    data = ticket_table.get_waste_year_month_days_town(code, year, month, db_connection)
    data['waste_qte'] = data.groupby(by='day')['waste_qte'].transform('mean')
    data.drop_duplicates(subset='day', inplace=True)
    data.sort_values(by='day', ascending=True, inplace=True)
    data.dropna(axis=0, inplace=True)
    days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    data['day'] = data['day'].astype(int)
    data['day'] = data['day'].apply(lambda x: days[x-1])
    json_data = {
        "labels": data['day'].astype(str).to_numpy(),
        "values": data['waste_qte'].to_numpy()
    }

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#get waste quantity by day of week for given  year and month  and town
@router.get(path="/statistics/unity/{code}/waste-quantity-day/{year}/{month}")
def get_day_waste_year_month_unity(code, year, month):
    ticket_table = TicketTable()
    data = ticket_table.get_waste_year_month_days_unity(code, year, month, db_connection)
    data['waste_qte'] = data.groupby(by='day')['waste_qte'].transform('mean')
    data.drop_duplicates(subset='day', inplace=True)
    data.sort_values(by='day', ascending=True, inplace=True)
    data.dropna(axis=0, inplace=True)
    days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    data['day'] = data['day'].astype(int)
    data['day'] = data['day'].apply(lambda x: days[x-1])
    json_data = {
        "labels": data['day'].astype(str).to_numpy(),
        "values": data['waste_qte'].to_numpy()
    }

    return json.dumps(json_data, cls=NumpyArrayEncoder)
