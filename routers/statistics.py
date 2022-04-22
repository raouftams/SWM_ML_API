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
async def get_town_information(id):
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
async def get_all_towns_information():
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
async def get_unity_information(code):
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
async def get_trend_by_town(town, period):
    data = {}
    if period == 'year':
        data = trend_by_town(365)
    if period == 'day':
        data = trend_by_town(1)
    if period == 'month':
        data = trend_by_town(30)

    #selecting the town
    town_data = data[town]
    town_data.index = town_data.index.strftime('%d-%m-%y')
    json_data ={
        "labels": town_data.index.to_numpy(),
        "values": town_data["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#get waste generation trend by period, town, and year
@router.get(path='/stats/trend/town/{town}/{period}/{year}')
async def get_trend_by_town_year(town, period, year):
    if period == 'year':
        data = trend_by_town_year(town, 182, int(year))
    if period == 'day':
        data = trend_by_town_year(town, 1, int(year))
    if period == 'month':
        data = trend_by_town_year(town, 30, int(year))

    data.index = data.index.strftime('%d-%m-%y')
    json_data ={
        "labels": data.index.to_numpy(),
        "values": data["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#get waste generation trend for all towns
@router.get(path="/stats/all-towns/trend/{period}")
async def get_trend(period):
    if period == 'year':
        data = all_towns_trend(365)
    if period == 'day':
        data = all_towns_trend(1)
    if period == 'month':
        data = all_towns_trend(30)

    #selecting the town
    data.index = data.index.strftime('%d-%m-%y')
    json_data ={
        "labels": data.index.to_numpy(),
        "values": data["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#get waste generation trend for all towns for one year
@router.get(path='/stats/all-towns/trend/{period}/{year}')
async def get_trend_year(period, year):
    if period == 'year':
        data = all_towns_trend_year(182, int(year))
    if period == 'day':
        data = all_towns_trend_year(1, int(year))
    if period == 'month':
        data = all_towns_trend_year(30, int(year))

    #selecting the town
    data.index = data.index.strftime('%d-%m-%y')
    json_data ={
        "labels": data.index.to_numpy(),
        "values": data["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#get waste generation trend for all towns
@router.get(path="/stats/trend/unity/{code}/{period}")
async def get_trend_by_unity(code, period):
    data = {}
    if period == 'year':
        data = trend_by_unity(365)
    if period == 'day':
        data = trend_by_unity(1)
    if period == 'month':
        data = trend_by_unity(30)

    #selecting the unity df
    df = data[code]
    df.index = df.index.strftime('%d-%m-%y')
    json_data = {
        "labels": df.index.to_numpy(),
        "values": df["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#get waste generation trend by period, town, and year
@router.get(path='/stats/trend/unity/{unity}/{period}/{year}')
async def get_trend_by_unity_year(unity, period, year):
    if period == 'year':
        data = trend_by_unity_year(unity, 182, int(year))
    if period == 'day':
        data = trend_by_unity_year(unity, 1, int(year))
    if period == 'month':
        data = trend_by_unity_year(unity, 30, int(year))

    data.index = data.index.strftime('%d-%m-%y')
    json_data ={
        "labels": data.index.to_numpy(),
        "values": data["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

"""-----------------------Waste Seasonality---------------------------"""

#get waste generation seaonality for all towns
@router.get(path="/stats/all-towns/seasonality/{period}")
async def get_seasonality(period):
    if period == 'year':
        data = all_towns_seasonality(365)
    if period == 'day':
        data = all_towns_seasonality(1)
    if period == 'month':
        data = all_towns_seasonality(30)

    #selecting the town
    data.index = data.index.strftime('%d-%m-%y')
    json_data ={
        "labels": data.index.to_numpy(),
        "values": data["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#get waste generation seasonality by (day, month, year) for town
@router.get(path="/stats/seasonality/town/{town}/{period}")
async def get_seasonality_by_town(town, period):
    data = {}
    if period == 'year':
        data = seasonality_by_town(365)
    if period == 'day':
        data = seasonality_by_town(1)
    if period == 'month':
        data = seasonality_by_town(30)

    #selecting the town
    town_data = data[town]
    town_data.index = town_data.index.strftime('%d-%m-%y')
    json_data ={
        "labels": town_data.index.to_numpy(),
        "values": town_data["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#get waste generation seasonality by (day, month, year) for unity
@router.get(path="/stats/seasonality/unity/{code}/{period}")
async def get_seasonality_by_unity(code, period):
    data = {}
    if period == 'year':
        data = seasonality_by_unity(365)
    if period == 'day':
        data = seasonality_by_unity(1)
    if period == 'month':
        data = seasonality_by_unity(30)

    #selecting the town
    town_data = data[code]
    town_data.index = town_data.index.strftime('%d-%m-%y')
    json_data ={
        "labels": town_data.index.to_numpy(),
        "values": town_data["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)


"""--------------------Rotations trend--------------------"""
#number of rotations trend for all towns
@router.get(path="/stats/rotations/all-towns/trend/{period}")
async def get_rotations_trend(period):
    if period == 'year':
        data = rotations_trend(365)
    if period == 'day':
        data = rotations_trend(1)
    if period == 'month':
        data = rotations_trend(30)

    #selecting the town
    data.index = data.index.strftime('%d-%m-%y')
    json_data ={
        "labels": data.index.to_numpy(),
        "values": data["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#number of rotations trend for all towns by year
@router.get(path="/stats/rotations/all-towns/trend/{period}/{year}")
async def get_rotations_trend_year(period, year):
    if period == 'year':
        data = rotations_trend_year(182, int(year))
    if period == 'day':
        data = rotations_trend_year(1, int(year))
    if period == 'month':
        data = rotations_trend_year(30, int(year))

    #selecting the town
    data.index = data.index.strftime('%d-%m-%y')
    json_data ={
        "labels": data.index.to_numpy(),
        "values": data["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#number of rotations trend by (day, month, year) for town
@router.get(path="/stats/rotations/trend/town/{town}/{period}")
async def get_rotations_trend_by_town(town, period):
    data = {}
    if period == 'year':
        data = rotations_trend_by_town(365)
    if period == 'day':
        data = rotations_trend_by_town(1)
    if period == 'month':
        data = rotations_trend_by_town(30)

    #selecting the town
    town_data = data[town]
    town_data.index = town_data.index.strftime('%d-%m-%y')
    json_data ={
        "labels": town_data.index.to_numpy(),
        "values": town_data["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#number of rotations trend by (day, month, year) for town by year
@router.get(path="/stats/rotations/trend/town/{town}/{period}/{year}")
async def get_rotations_trend_by_town_year(town, period, year):
    data = {}
    if period == 'year':
        data = rotations_trend_by_town_year(town, 182, int(year))
    if period == 'day':
        data = rotations_trend_by_town_year(town, 1, int(year))
    if period == 'month':
        data = rotations_trend_by_town_year(town, 30, int(year))

    data.index = data.index.strftime('%d-%m-%y')
    json_data ={
        "labels": data.index.to_numpy(),
        "values": data["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#number of rotations trend by (day, month, year) for a unity
@router.get(path="/stats/rotations/trend/unity/{code}/{period}")
async def get_rotations_trend_by_unity(code, period):
    data = {}
    if period == 'year':
        data = rotations_trend_by_unity(365)
    if period == 'day':
        data = rotations_trend_by_unity(1)
    if period == 'month':
        data = rotations_trend_by_unity(30)

    #selecting the unity df
    df = data[code]
    df.index = df.index.strftime('%d-%m-%y')
    json_data = {
        "labels": df.index.to_numpy(),
        "values": df["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#number of rotations trend by (day, month, year) for unity by year
@router.get(path="/stats/rotations/trend/unity/{unity}/{period}/{year}")
async def get_rotations_trend_by_unity_year(unity, period, year):
    data = {}
    if period == 'year':
        data = rotations_trend_by_unity_year(unity, 182, int(year))
    if period == 'day':
        data = rotations_trend_by_unity_year(unity, 1, int(year))
    if period == 'month':
        data = rotations_trend_by_unity_year(unity, 30, int(year))

    data.index = data.index.strftime('%d-%m-%y')
    json_data ={
        "labels": data.index.to_numpy(),
        "values": data["values"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#number of rotations trend by (day, month, year) for town
@router.get(path="/stats/rotations/trend/all-towns/hour")
async def get_rotations_trend_hour():
    data = rotations_trend_by_hour()
    json_data ={
        "labels": data['heure'].astype(str).to_numpy(),
        "values": data["rotations"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#number of rotations trend by (day, month, year) for town
@router.get(path="/stats/rotations/trend/hour/town/{town}")
async def get_rotations_trend_by_town_hour(town):
    data = rotations_trend_by_town_hour(town)
    json_data ={
        "labels": data['heure'].astype(str).to_numpy(),
        "values": data["rotations"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

#number of rotations trend by (day, month, year) for unity
@router.get(path="/stats/rotations/trend/hour/unity/{unity}")
async def get_rotations_trend_by_unity_hour(unity):
    data = rotations_trend_by_unity_hour(unity)
    json_data ={
        "labels": data['heure'].astype(str).to_numpy(),
        "values": data["rotations"].to_numpy()
    } 

    return json.dumps(json_data, cls=NumpyArrayEncoder)

"""-----------------Holidays data-------------------------"""

#get waste quantity by hijri holidays
@router.get(path="/stats/waste/holidays/town/{town}")
async def get_holiday_waste_by_town(town):
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
async def get_holiday_waste_by_unity(code):
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
async def get_holiday_waste_qte():
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
async def get_season_waste_by_town(town):
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
async def get_season_waste_by_unity(code):
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
async def get_season_waste_qte():
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
async def get_qte_change_rate_by_season():
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


