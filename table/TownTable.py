from table.Table import Table

class TownTable(Table):

    def __init__(self) -> None:
        super().__init__("commune")
    

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
            cursor.execute("SELECT code, name, ST_AsGeoJson(ST_Transform(ST_SetSRID(geom,32631), 4326))::jsonb as geom FROM town_test where code = 'C001'")
            #get result
            result = cursor.fetchall()
            #close cursor
            cursor.close()
            #check if result is empty
            if result != []:
                return result

            return None

    #check if town name exists
    def exists_name(self, name, db_connection):
        """
        Args
            name: town name
            db_connection: pysoppg2 instance
        purpose
            check if town name exists in database
        """
        if db_connection != None:
            #create cursor
            cursor = db_connection.cursor()

            #execute query
            cursor.execute("SELECT * FROM commune WHERE name = '{}'".format(name.upper()))
            #get result
            result = cursor.fetchall()
            #close cursor
            cursor.close()
            #check if result is empty
            if result != []:
                return True

            return False

    #get code from name
    def get_code_from_name(self, name, db_connection):
        """
        Args:
            name: town name
            db_connection: psycopg2 db connection instance
        Get town's code using town's name
        """
        if db_connection != None:
            #create cursor
            cursor = db_connection.cursor()
            #execute query
            cursor.execute("SELECT code FROM commune WHERE name = '{}'".format(name.upper()))
            #get result
            result = cursor.fetchone()
            #close cursor
            cursor.close()
            #check if result is empty
            if result != []:
                return result[0]
            
            return None