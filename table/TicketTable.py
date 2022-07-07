from table.Table import Table
import pandas as pd
class TicketTable(Table):

    def __init__(self) -> None:
        super().__init__("ticket")
    
    #get total waste quantity by town 
    def get_total_waste_quantity(self, code, db_connection):
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
            cursor.execute("SELECT sum(t.net) from ticket t, rotation r where t.code = r.code_ticket and t.date = r.date and t.cet = r.cet and r.code_town = '{}'".format(code))
            #get result
            result = cursor.fetchone()
            #close cursor
            cursor.close()
            #check if result is empty
            if result != []:
                return result
        return None
    

    #get total waste quantity by unity 
    def get_unity_total_waste_quantity(self, code, db_connection):
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
            cursor.execute("SELECT sum(t.net) from ticket t, rotation r where t.code = r.code_ticket and t.date = r.date and t.cet = r.cet and r.code_town in (select code from commune where code_unity = '{}')".format(code))
            #get result
            result = cursor.fetchone()
            #close cursor
            cursor.close()
            #check if result is empty
            if result != []:
                return result
        return None

    #get all towns waste quantity by hour for given year and month
    def get_all_towns_waste_year_month(self, year, month, db_connection):
        sql = 'select sum(t.net)/1000 as waste_qte, t.heure from ticket t  where Extract(Year from t.date) = {} and Extract(Month from t.date) = {} group by t.heure'.format(year, month)
        return pd.read_sql_query(sql, db_connection)
    
    #get waste quantity by hour for given year and month and town
    def get_all_towns_waste_year_month_town(self, code, year, month, db_connection):
        sql = "select sum(t.net)/1000 as waste_qte, t.heure from ticket t, rotation r where r.code_ticket = t.code and r.date = t.date and r.code_town = '{}' and Extract(Year from t.date) = {} and Extract(Month from t.date) = {} group by t.heure".format(code, year, month)
        return pd.read_sql_query(sql, db_connection)
    
    #get waste quantity by hour for given year and month and unity
    def get_all_towns_waste_year_month_unity(self, code, year, month, db_connection):
        sql = "select sum(t.net)/1000 as waste_qte, t.heure from ticket t, rotation r, commune c where r.code_ticket = t.code and r.date = t.date and c.code = r.code_town and c.code_unity = '{}' and Extract(Year from t.date) = {} and Extract(Month from t.date) = {} group by t.heure".format(code, year, month)
        return pd.read_sql_query(sql, db_connection)
    
    #get all towns waste quantity by day of week
    def get_all_towns_waste_year_month_days(self, year, month, db_connection):
        sql = "select sum(t.net)/1000 as waste_qte, extract(isodow from t.date) as day from ticket t where Extract(Year from t.date) = {} and Extract(Month from t.date) = {} group by t.date".format(year, month)
        return pd.read_sql_query(sql, db_connection)
    
    #get waste quantity by day of week for a given town 
    def get_waste_year_month_days_town(self, code, year, month, db_connection):
        sql = "select sum(t.net)/1000 as waste_qte, extract(isodow from t.date) as day from ticket t, rotation r where r.date = t.date and r.code_ticket = t.code and r.code_town = '{}' and Extract(Year from t.date) = {} and Extract(Month from t.date) = {} group by t.date".format(code, year, month)
        return pd.read_sql_query(sql, db_connection)
    
    #get waste quantity by day of week for a given unity 
    def get_waste_year_month_days_unity(self, code, year, month, db_connection):
        sql = "select sum(t.net)/1000 as waste_qte, extract(isodow from t.date) as day from ticket t, rotation r, commune c where r.date = t.date and r.code_ticket = t.code and r.code_town = c.code and c.code_unity = '{}' and Extract(Year from t.date) = {} and Extract(Month from t.date) = {} group by t.date".format(code, year, month)
        return pd.read_sql_query(sql, db_connection)

    #get total waste quantity  
    def get_total_waste(self, db_connection):
        """
        Args:
            db_connection: psycopg2 db connection instance
        Get all towns
        """
        if db_connection != None:
            #create cursor
            cursor = db_connection.cursor()
            #execute query
            cursor.execute("SELECT sum(net) from ticket")
            #get result
            result = cursor.fetchone()
            #close cursor
            cursor.close()
            #check if result is empty
            if result != []:
                return result
        return None