from table.Table import Table 

class RotationTable(Table):

    def __init__(self) -> None:
        super().__init__("rotation")
    
    #get number of rotations by town 
    def get_town_nb_rotations(self, town_code, db_connection):
        """
        Args:
            town_code: town code
            db_connection: psycopg2 db connection instance
        Get all towns
        """
        if db_connection != None:
            #create cursor
            cursor = db_connection.cursor()
            #execute query
            cursor.execute("SELECT count(*) from rotation where code_town = '{}'".format(town_code))
            #get result
            result = cursor.fetchone()
            #close cursor
            cursor.close()
            #check if result is empty
            if result != []:
                return result
        return None

    #get number of rotations 
    def get_nb_rotations(self, db_connection):
        """
        Args:
            db_connection: psycopg2 db connection instance
        Get all towns
        """
        if db_connection != None:
            #create cursor
            cursor = db_connection.cursor()
            #execute query
            cursor.execute("SELECT count(*) from rotation")
            #get result
            result = cursor.fetchone()
            #close cursor
            cursor.close()
            #check if result is empty
            if result != []:
                return result
        return None


    #get number of daily rotations by unity
    def get_town_nb_rotations_day(self, code, db_connection):
        """
        Args:
            code: unity code
            db_connection: psycopg2 db connection instance
        Get all towns
        """
        nb_total_rotations = self.get_town_nb_rotations(code, db_connection)[0]
        if db_connection != None:
            #create cursor
            cursor = db_connection.cursor()
            #execute query
            cursor.execute("select max(date)-min(date) as days from rotation")
            #get result
            result = cursor.fetchone()
            #close cursor
            cursor.close()
            #check if result is empty
            if result != []:
                return [nb_total_rotations/result[0]]
        return None

    #get number of rotations by unity
    def get_unity_nb_rotations(self, code, db_connection):
        """
        Args:
            town_code: town code
            db_connection: psycopg2 db connection instance
        Get all towns
        """
        if db_connection != None:
            #create cursor
            cursor = db_connection.cursor()
            #execute query
            cursor.execute("SELECT count(*) from rotation where code_town in (select code from commune where code_unity = '{}')".format(code))
            #get result
            result = cursor.fetchone()
            #close cursor
            cursor.close()
            #check if result is empty
            if result != []:
                return result
        return None

    #get number of daily rotations by unity
    def get_unity_nb_rotations_day(self, code, db_connection):
        """
        Args:
            code: unity code
            db_connection: psycopg2 db connection instance
        Get all towns
        """
        nb_total_rotations = self.get_unity_nb_rotations(code, db_connection)[0]
        if db_connection != None:
            #create cursor
            cursor = db_connection.cursor()
            #execute query
            cursor.execute("select max(date)-min(date) as days from rotation")
            #get result
            result = cursor.fetchone()
            #close cursor
            cursor.close()
            #check if result is empty
            if result != []:
                return [nb_total_rotations/result[0]]
        return None

    #get number of used vehicles by town 
    def get_town_used_vehicles(self, town_code, db_connection):
        """
        Args:
            town_code: town code
            db_connection: psycopg2 db connection instance
        Get all towns
        """
        if db_connection != None:
            #create cursor
            cursor = db_connection.cursor()
            #execute query
            cursor.execute("SELECT count(DISTINCT id_vehicle) from rotation where code_town = '{}'".format(town_code))
            #get result
            result = cursor.fetchone()
            #close cursor
            cursor.close()
            #check if result is empty
            if result != []:
                return result
        return None


    #get number of used vehicles by town 
    def get_unity_used_vehicles(self, code, db_connection):
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
            cursor.execute("SELECT count(DISTINCT id_vehicle) from rotation where code_town in (select code from commune where code_unity = '{}')".format(code))
            #get result
            result = cursor.fetchone()
            #close cursor
            cursor.close()
            #check if result is empty
            if result != []:
                return result
        return None

    def exists(self, id, db_connection):
        """
        Arguments: 
            id: rotation id (primary key)
            db_connection: psycopg2 db connection instance
        """
        #create a cursor
        cursor = db_connection.cursor()
        #execute query
        cursor.execute("SELECT * from {} where id = {}".format(self.table_name, id))
        #get selected records
        data = cursor.fetchall()
        #close cursor
        cursor.close()

        if data == []:
            return False
        
        return True