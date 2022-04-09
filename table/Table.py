from psycopg2.extensions import AsIs

class Table:

    def __init__(self, table_name) -> None:
        self.table_name = table_name.lower()
    

    def get_all(self, db_connection):
        #create a cursor
        cursor = db_connection.cursor()
        #execute query
        cursor.execute("Select * from {}".format(self.table_name))
        #get data
        data = cursor.fetchall()
        #close cursor
        cursor.close()

        return data

    def insert(self, data, db_connection):
        """
        Arguments:
            data: dict 
            db_connection: psycopg2 db connection instance
        """
        #create a cursor
        cursor = db_connection.cursor()
        #create statement
        statement = "INSERT INTO {}(%s) VALUES %s".format(self.table_name)
        #execute statement
        cursor.execute(statement, (AsIs(','.join(data.keys())), tuple(data.values())))
        #commit changes
        db_connection.commit()
        #close cursor
        cursor.close()

    def exists(self, code, db_connection):
        """
        Arguments: 
            code: vehicle code (primary key)
            db_connection: psycopg2 db connection instance
        """
        #create a cursor
        cursor = db_connection.cursor()
        #execute query
        cursor.execute("SELECT * from {} where code = '{}'".format(self.table_name, code))
        #get selected records
        data = cursor.fetchall()
        #close cursor
        cursor.close()

        if data == []:
            return False
        
        return True