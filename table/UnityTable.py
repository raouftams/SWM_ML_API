from table.Table import Table

class UnityTable(Table):

    def __init__(self) -> None:
        super().__init__("unity")
    
    #get all codes, names and geom as GeoJSON
    def get_all(self, db_connection):
        """
        Args:
            db_connection: psycopg2 db connection instance
        Get all towns
        """
        if db_connection != None:
            #create cursor
            cursor = db_connection.cursor()
            #execute query
            cursor.execute("SELECT code, name, ST_AsGeoJson(ST_Transform(ST_SetSRID(geom,32631), 4326))::jsonb as geom FROM unity where geom !=''")
            #get result
            result = cursor.fetchall()
            #close cursor
            cursor.close()
            #check if result is empty
            if result != []:
                return result

            return None