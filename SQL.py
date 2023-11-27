import pyodbc 
import sqlalchemy as sql
import mysql.connector
import pandas as pd


server = "sql5.freesqldatabase.com"
database = "sql5664042"
username = "sql5664042"
password = "mwQUwFLr4D"
port = "3306"
text = ""
response = ""

connection = mysql.connector.connect(user=username,password=password,host=server,database=database)
cursor = connection.cursor()
cursor.execute("SELECT * FROM Chatbot")
cursor.execute("INSERT INTO Chatbot VALUES(%s,%s)",(text,response))
connection.commit()

cursor.fetchall()
cursor.close()
connection.close()

