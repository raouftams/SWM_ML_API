from table.Table import Table
import pandas as pd
class VehicleTable(Table):

    def __init__(self) -> None:
        super().__init__("vehicle")
    

    #get all vehicles codes and matricules
    def get_all(self, db_connection):
        if db_connection != None:
            cursor = db_connection.cursor()
            cursor.execute("select code, ancien_matricule, nouveau_matricule from vehicle")
            result = cursor.fetchall()
            cursor.close()
            return result

    #check if matricule exists in database
    def exists_mat(self, mat, db_connection):
        """
        Args: mat: vehicle matricule
            db_connection: psycopg2 db instance
        puprose: 
            check if mat exists in vehicle table
        """
        if db_connection != None:
            #create cursor
            cursor = db_connection.cursor()
            """check if mat = nouveau_matricule"""
            #execute query
            cursor.execute("SELECT * FROM vehicle WHERE nouveau_matricule = '{}'".format(mat))
            #get result
            result = cursor.fetchall()
            #check if result is empty
            if result != []:
                cursor.close()
                return True
            else:
                """check if mat = ancien_matricule"""
                cursor.execute("SELECT * FROM vehicle WHERE ancien_matricule = '{}'".format(mat))
                result = cursor.fetchall()
                cursor.close()
                if result != []:
                    return True

            return False

    #get code from matricule
    def get_code_from_mat(self, mat, db_connection):
        """
        Args:
            mat: vehicle matricule
            db_connection: psycopg2 db connection instance
        Get vehicle's code using vehicle' matricule
        """
        if db_connection != None:
            #create cursor
            cursor = db_connection.cursor()
            #check with nouveau_matricule
            #execute query
            cursor.execute("SELECT code FROM vehicle WHERE nouveau_matricule = '{}'".format(mat))
            #get result
            result = cursor.fetchone()
            #check if result is empty
            if result != []:
                cursor.close()
                return result[0]
            else:
                #check with ancien_matricule
                cursor.execute("SELECT code FROM vehicle WHERE ancien_matricule = '{}'".format(mat))
                result = cursor.fetchone()
                cursor.close()
                if result != []:
                    return result[0]
            
            return None

    #get number of vehicles
    def get_nb_vehicles(self, db_connection):
        """
        Args:
            db_connection: psycopg2 db connection instance
        Get all towns
        """
        if db_connection != None:
            #create cursor
            cursor = db_connection.cursor()
            #execute query
            cursor.execute("SELECT count(*) from vehicle") 
            #get result
            result = cursor.fetchone()
            #close cursor
            cursor.close()
            #check if result is empty
            if result != []:
                return result
        return None

    #get number of owned vehicles by town
    def nb_owned_vehicles(self, code, db_connection):
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
            cursor.execute("SELECT count(*) from vehicle where code_commune = '{}'".format(code))
            #get result
            result = cursor.fetchone()
            #close cursor
            cursor.close()
            #check if result is empty
            if result != []:
                return result
        return None

    def get_efficiency_by_mark_year_month(self, year, month, db_connection):
        sql = 'select v.marque, v.code, avg(t.net/(v.volume*1000)) as compact_rate from vehicle v, ticket t, rotation r  where t.code = r.code_ticket and t.date = r.date and r.id_vehicle = v.code and v.volume != 0  and Extract(Year from t.date) = {} and Extract(Month from t.date) = {} group by (v.marque, v.code)'.format(year, month)
        return pd.read_sql_query(sql, db_connection)
    
    def get_efficiency_by_mark_year_month_town(self, code, year, month, db_connection):
        sql = "select v.marque, v.code, avg(t.net/(v.volume*1000)) as compact_rate from vehicle v, ticket t, rotation r where r.code_town = '{}' and t.code = r.code_ticket and t.date = r.date and r.id_vehicle = v.code and v.volume != 0  and Extract(Year from t.date) = {} and Extract(Month from t.date) = {} group by (v.marque, v.code)".format(code, year, month)
        return pd.read_sql_query(sql, db_connection)
    
    def get_efficiency_by_mark_year_month_unity(self, code, year, month, db_connection):
        sql = "select v.marque, v.code, avg(t.net/(v.volume*1000)) as compact_rate from vehicle v, ticket t, rotation r, commune c where r.code_town = c.code and c.code_unity = '{}' and t.code = r.code_ticket and t.date = r.date and r.id_vehicle = v.code and v.volume != 0  and Extract(Year from t.date) = {} and Extract(Month from t.date) = {} group by (v.marque, v.code)".format(code, year, month)
        return pd.read_sql_query(sql, db_connection)
    

    def get_efficiency_by_volume_year_month(self, year, month, db_connection):
        sql = 'select v.volume, v.code, avg(t.net/(v.volume*1000)) as compact_rate from vehicle v, ticket t, rotation r  where t.code = r.code_ticket and t.date = r.date and r.id_vehicle = v.code and v.volume != 0  and Extract(Year from t.date) = {} and Extract(Month from t.date) = {} group by (v.volume, v.code)'.format(year, month)
        return pd.read_sql_query(sql, db_connection)
    
    def get_efficiency_by_volume_year_month_town(self, code, year, month, db_connection):
        sql = "select v.volume, v.code, avg(t.net/(v.volume*1000)) as compact_rate from vehicle v, ticket t, rotation r  where r.code_town = '{}' and t.code = r.code_ticket and t.date = r.date and r.id_vehicle = v.code and v.volume != 0  and Extract(Year from t.date) = {} and Extract(Month from t.date) = {} group by (v.volume, v.code)".format(code, year, month)
        return pd.read_sql_query(sql, db_connection)
    
    def get_efficiency_by_volume_year_month_unity(self, code, year, month, db_connection):
        sql = "select v.volume, v.code, avg(t.net/(v.volume*1000)) as compact_rate from vehicle v, ticket t, rotation r, commune c  where r.code_town = c.code and c.code_unity = '{}' and t.code = r.code_ticket and t.date = r.date and r.id_vehicle = v.code and v.volume != 0  and Extract(Year from t.date) = {} and Extract(Month from t.date) = {} group by (v.volume, v.code)".format(code, year, month)
        return pd.read_sql_query(sql, db_connection)