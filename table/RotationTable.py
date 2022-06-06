from table.Table import Table 
import pandas as pd

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

    #get number of rotations by town and date
    def nb_rotations_by_town_date(self, db_connection):
        """
        Args:
            db_connection: psycopg2 db connection instance
        Get all towns
        """
        if db_connection != None:
            #create cursor
            cursor = db_connection.cursor()
            #execute query
            cursor.execute("SELECT code_town, date, count(*) from rotation group by(code_town, date)")
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


    #get compaction rate by hours and towns
    def get_compaction_rate_hour(self, db_connection):
        sql = 'select (t.net/(v.volume*1000)) as taux_compaction, r.heure, r.code_town, c.code_unity from ticket t, vehicle v, rotation r, commune c where c.code = r.code_town and v.volume != 0 and r.id_vehicle = v.code and r.code_ticket = t.code'
        return pd.read_sql_query(sql, db_connection)

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


    #get years and months registred
    def get_years_and_months(self, db_connection):
        sql = 'select distinct(Extract(Year from t.date)) as "year", Extract(Month from t.date) as "month" from rotation t '
        return pd.read_sql_query(sql, db_connection)

    #get waste qte by year and month for all towns
    def get_waste_qte_by_year_month_towns(self, year, month, db_connection):
        sql = 'select sum(t.net)/1000 as waste_qte, r.code_town from ticket t, rotation r where r.code_ticket = t.code and r.date = t.date and Extract(Year from t.date) = {} and Extract(Month from t.date) = {} group by r.code_town'.format(year, month)
        return pd.read_sql_query(sql, db_connection)

    #get waste qte by year and month for all towns
    def get_waste_qte_by_year_month_unities(self, year, month, db_connection):
        sql = 'select sum(t.net)/1000 as waste_qte, c.code_unity from ticket t, rotation r, commune c where c.code = r.code_town and r.code_ticket = t.code and r.date = t.date and Extract(Year from t.date) = {} and Extract(Month from t.date) = {} group by c.code_unity'.format(year, month)
        return pd.read_sql_query(sql, db_connection)

    #get efficiency by year and month for all unities
    def get_efficiency_by_year_month_unities(self, year, month, db_connection):
        sql = 'select avg((t.net/(v.volume*1000))) as efficiency, c.code_unity from ticket t, rotation r, commune c, vehicle v where r.id_vehicle = v.code and v.volume != 0 and c.code = r.code_town and r.code_ticket = t.code and r.date = t.date and Extract(Year from t.date) = {} and Extract(Month from t.date) = {} group by c.code_unity'.format(year, month)
        return pd.read_sql_query(sql, db_connection)

    #get efficiency by year and month for all towns
    def get_efficiency_by_year_month_towns(self, year, month, db_connection):
        sql = 'select avg((t.net/(v.volume*1000))) as efficiency, r.code_town from ticket t, rotation r, vehicle v where r.id_vehicle = v.code and v.volume != 0 and r.code_ticket = t.code and r.date = t.date and Extract(Year from t.date) = {} and Extract(Month from t.date) = {} group by r.code_town'.format(year, month)
        return pd.read_sql_query(sql, db_connection)


    #get information by year and month for all towns
    def get_info_by_year_month_all_towns(self, year, month, db_connection):
        sql = 'select  count(r.*)/30 as nb_rotations_day, count(r.*) as nb_rotations, count(distinct(r.id_vehicle)) as nb_vehicles,  sum(t.net)/1000 as waste_qte,  (sum(t.net)/30)/1000 as waste_qte_day, avg((t.net/(v.volume*1000))) as compact_rate from rotation r, ticket t, commune c, vehicle v  where v.code = r.id_vehicle and v.volume != 0 and c.code = r.code_town and t.code = r.code_ticket  and t.date = r.date and Extract(Year from r.date) = {} and Extract(Month from r.date) = {}'.format(year, month)
        return pd.read_sql_query(sql, db_connection)

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