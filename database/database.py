from configparser import ConfigParser
import psycopg2

def connect():
    """credits to: https://www.postgresqltutorial.com/postgresql-python/connect/"""
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(host="localhost", database="swm_ml", user="username", password="password")
        return conn

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
