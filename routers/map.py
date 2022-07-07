from fastapi import APIRouter
import json
from database.database import connect
from pattern_recognition import month_clustering, year_clustering, week_clustering
from table.RotationTable import RotationTable
from table.TownTable import TownTable
from table.UnityTable import UnityTable
from utilities import NumpyArrayEncoder

router = APIRouter()


db_connection = connect()

@router.get("/all-towns", tags=["map"])
async def all_towns():
    town_table = TownTable()
    towns = town_table.get_all(db_connection)
    features = []
    for row in towns:
        features.append({
            "type": "Feature",
            "properties": {"code": row[0], "name":row[1], "superficie": row[2], "densité": row[3]},
            "geometry": row[4] 
        })
    feature_collection = {"type": "FeatureCollection", "features": features}
    return(feature_collection)

@router.get("/all-towns/clustering")
async def all_towns_clustering():
    month_data = month_clustering()
    year_data = year_clustering()
    week_data = week_clustering()
    
    json_data = {
        "month": {
            "towns": list(month_data.keys()),
            "labels": list(month_data.values())
        },
        "year": {
            "towns": list(year_data.keys()),
            "labels": list(year_data.values())
        },
        "week": {
            "towns": list(week_data.keys()),
            "labels": list(week_data.values())
        }
    }

    return json.dumps(json_data, cls=NumpyArrayEncoder)


@router.get("/town/{id}")
async def get_town(id):
    town_table = TownTable()
    town = town_table.get_one(id, db_connection)
    features = []
    features=[{
            "type": "Feature",
            "properties": {"code": town[0], "name":town[1]},
            "geometry": town[2] 
    }]
    feature_collection = {"type": "FeatureCollection", "features": features}
    return(feature_collection)


@router.get("/all-unities", tags=['map'])
async def all_unities():
    unity_table = UnityTable()
    data = unity_table.get_all(db_connection)
    features = []
    for row in data:
        features.append({
            "type": "Feature",
            "properties": {"code": row[0], "name":row[1]},
            "geometry": row[2] 
        })
    feature_collection = {"type": "FeatureCollection", "features": features}
    return(feature_collection)

@router.get(path="/all-towns/waste-qte/{year}/{month}")
async def all_towns_waste_qte(year, month):
    rotation_table = RotationTable()
    data = rotation_table.get_waste_qte_by_year_month_towns(year, month, db_connection)
    data.fillna(0, axis=0, inplace=True)
    json_data = {
        'values': data['waste_qte'].to_numpy(),
        'regions': data['code_town'].to_numpy()
    }

    return json.dumps(json_data, cls=NumpyArrayEncoder)

@router.get(path="/all-unities/waste-qte/{year}/{month}")
async def all_unities_waste_qte(year, month):
    rotation_table = RotationTable()
    data = rotation_table.get_waste_qte_by_year_month_unities(year, month, db_connection)
    data.fillna(0, axis=0, inplace=True)
    json_data = {
        'values': data['waste_qte'].to_numpy(),
        'regions': data['code_unity'].to_numpy()
    }

    return json.dumps(json_data, cls=NumpyArrayEncoder)

@router.get(path="/all-towns/efficiency/{year}/{month}")
async def all_towns_waste_qte(year, month):
    rotation_table = RotationTable()
    data = rotation_table.get_efficiency_by_year_month_towns(year, month, db_connection)
    data.fillna(0, axis=0, inplace=True)
    json_data = {
        'values': data['efficiency'].to_numpy(),
        'regions': data['code_town'].to_numpy()
    }

    return json.dumps(json_data, cls=NumpyArrayEncoder)

@router.get(path="/all-unities/efficiency/{year}/{month}")
async def all_unities_waste_qte(year, month):
    rotation_table = RotationTable()
    data = rotation_table.get_efficiency_by_year_month_unities(year, month, db_connection)
    data.fillna(0, axis=0, inplace=True)
    json_data = {
        'values': data['efficiency'].to_numpy(),
        'regions': data['code_unity'].to_numpy()
    }

    return json.dumps(json_data, cls=NumpyArrayEncoder)