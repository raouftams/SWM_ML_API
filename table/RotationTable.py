from table.Table import Table 

class RotationTable(Table):

    def __init__(self) -> None:
        super().__init__("rotation")
    

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