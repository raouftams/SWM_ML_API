from table.Table import Table

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


