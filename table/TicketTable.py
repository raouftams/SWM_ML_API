from table.Table import Table

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