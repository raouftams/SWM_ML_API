from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
from database.database import connect
from table.TownTable import TownTable
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_connection = connect()

@app.get("/all-towns")
def all_towns():
    town_table = TownTable()
    towns = town_table.get_all(db_connection)
    features = []
    for row in towns:
        features.append({
            "type": "Feature",
            "properties": {"code": row[0], "name":row[1]},
            "geometry": row[2] 
        })
    feature_collection = {"type": "FeatureCollection", "features": features}
    return(feature_collection)

@app.get("/izzan")
def izzan():
    return "izzan"