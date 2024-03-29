from table.Table import Table

class TownTable(Table):

    def __init__(self) -> None:
        super().__init__("commune")
    
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
            cursor.execute("SELECT code, name, superficie, densite, ST_AsGeoJson(ST_Transform(ST_SetSRID(geom,32631), 4326))::jsonb as geom FROM commune where geom !=''")
            #get result
            result = cursor.fetchall()
            #close cursor
            cursor.close()
            #check if result is empty
            if result != []:
                return result

            return None
        

    #get name and geom as GeoJSON by code
    def get_one(self, code, db_connection):
        """
        Args:
            code: town code
            db_connection: psycopg2 db connection instance
        Get all towns
        """
        if db_connection != None:
            #create cursor
            cursor = db_connection.cursor()
            #execute query
            cursor.execute("SELECT code, name, ST_AsGeoJson(ST_Transform(ST_SetSRID(geom,32631), 4326))::jsonb as geom FROM commune where code = '{}'".format(code))
            #get result
            result = cursor.fetchone()
            #close cursor
            cursor.close()
            #check if result is empty
            if result != []:
                return result
        return None

    def get_population(self, code, db_connection):
        """
        Args:
            code: town code
            db_connection: psycopg2 db connection instance
        Get all towns
        """
        if db_connection != None:
            #create cursor
            cursor = db_connection.cursor()
            #execute query
            cursor.execute("SELECT pop2020 FROM commune where code = '{}'".format(code))
            #get result
            result = cursor.fetchone()
            #close cursor
            cursor.close()
            #check if result is empty
            if result != []:
                return result
        return None

    def get_ratio(self, code, db_connection):
        """
        Args:
            code: town code
            db_connection: psycopg2 db connection instance
        Get all towns
        """
        if db_connection != None:
            #create cursor
            cursor = db_connection.cursor()
            #execute query
            cursor.execute("SELECT ratio FROM commune where code = '{}'".format(code))
            #get result
            result = cursor.fetchone()
            #close cursor
            cursor.close()
            #check if result is empty
            if result != []:
                return result
        return None

    def get_unity_population(self, code, db_connection):
        """
        Args:
            code: unity code
            db_connection: psycopg2 db connection instance
        Get all towns
        """
        if db_connection != None:
            #create cursor
            cursor = db_connection.cursor()
            #execute query
            cursor.execute("SELECT sum(pop2020) FROM commune where code_unity = '{}'".format(code))
            #get result
            result = cursor.fetchone()
            #close cursor
            cursor.close()
            #check if result is empty
            if result != []:
                return result
        return None

    def get_unity_ratio(self, code, db_connection):
        """
        Args:
            code: town code
            db_connection: psycopg2 db connection instance
        Get all towns
        """
        if db_connection != None:
            #create cursor
            cursor = db_connection.cursor()
            #execute query
            cursor.execute("SELECT sum(ratio)/count(*) FROM commune where code_unity = '{}'".format(code))
            #get result
            result = cursor.fetchone()
            #close cursor
            cursor.close()
            #check if result is empty
            if result != []:
                return result
        return None

    def get_population_year(self, year, db_connection):
        if db_connection != None:
            #create cursor
            cursor = db_connection.cursor()
            #execute query
            sql = 'select sum(pop2016) from commune'
            if(year == 2017):
                sql = 'select sum(pop2017) from commune'
            if(year == 2018):
                sql = 'select sum(pop2018) from commune'
            if(year == 2019):
                sql = 'select sum(2019) from commune'
            if(year == 2020):
                sql = 'select sum(pop2020) from commune'
            if(year == 2021):
                sql = 'select sum(pop2021) from commune'
            if(year == 2022):
                sql = 'select sum(pop2022) from commune'

            cursor.execute(sql)
            #get result
            result = cursor.fetchone()
            #close cursor
            cursor.close()
            #check if result is empty
            if result != []:
                return result
        return None
    
    def get_population_year_town(self, code, year, db_connection):
        if db_connection != None:
            #create cursor
            cursor = db_connection.cursor()
            #execute query
            sql = "select pop2016 from commune where code = '{}'".format(code)
            if(year == 2017):
                sql = "select pop2017 from commune where code = '{}'".format(code)
            if(year == 2018):
                sql = "select pop2018 from commune where code = '{}'".format(code)
            if(year == 2019):
                sql = "select 2019 from commune where code = '{}'".format(code)
            if(year == 2020):
                sql = "select pop2020 from commune where code = '{}'".format(code)
            if(year == 2021):
                sql = "select pop2021 from commune where code = '{}'".format(code)
            if(year == 2022):
                sql = "select pop2022 from commune where code = '{}'".format(code)

            cursor.execute(sql)
            #get result
            result = cursor.fetchone()
            #close cursor
            cursor.close()
            #check if result is empty
            if result != []:
                return result
        return None

    def get_population_year_unity(self, code, year, db_connection):
        if db_connection != None:
            #create cursor
            cursor = db_connection.cursor()
            #execute query
            sql = "select sum(pop2016) from commune where code_unity = '{}'".format(code)
            if(year == 2017):
                sql = "select sum(pop2017) from commune where code_unity = '{}'".format(code)
            if(year == 2018):
                sql = "select sum(pop2018) from commune where code_unity = '{}'".format(code)
            if(year == 2019):
                sql = "select sum(2019) from commune where code_unity = '{}'".format(code)
            if(year == 2020):
                sql = "select sum(pop2020) from commune where code_unity = '{}'".format(code)
            if(year == 2021):
                sql = "select sum(pop2021) from commune where code_unity = '{}'".format(code)
            if(year == 2022):
                sql = "select sum(pop2022) from commune where code_unity = '{}'".format(code)

            cursor.execute(sql)
            #get result
            result = cursor.fetchone()
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