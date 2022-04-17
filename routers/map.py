from fastapi import APIRouter
import json
from database.database import connect
from table.TownTable import TownTable
from table.UnityTable import UnityTable

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