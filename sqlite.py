import sqlite3

## connect to sqllite
connection=sqlite3.connect("student.db")

##create a cursor object to insert record,create table
cursor=connection.cursor()

## create the table
table_info="""
create table STUDENT(NAME VARCHAR(25),CLASS VARCHAR(25),
SECTION VARCHAR(25),MARKS INT)
"""

cursor.execute(table_info)

## Insert some more records
cursor.execute('''Insert Into STUDENT values('Robin','Robotics','A',90)''')
cursor.execute('''Insert Into STUDENT values('Dilip','Machine Learning','B',100)''')
cursor.execute('''Insert Into STUDENT values('Badal','Deeplearning','A',86)''')
cursor.execute('''Insert Into STUDENT values('Rawat','MLOPS','A',50)''')
cursor.execute('''Insert Into STUDENT values('Seli','Gen AI','A',35)''')

## Display all the records
print("The inserted records are")
data=cursor.execute('''Select * from STUDENT''')
for row in data:
    print(row)

## Commit your changes in the database
connection.commit()
connection.close()
