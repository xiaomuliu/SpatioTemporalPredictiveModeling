#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
====================================
Pull Crime Data From Datawarehouse
====================================
Created on Fri Sep 30 14:52:50 2016

@author: xiaomuliu
"""


import cx_Oracle, csv
from FetchDBData import fetch_data_to_dataframe 

file_path = './CrimeData/DWH/'
crime_types = ["Homicide","SexualAssault","Robbery","AggAssaultBattery","SimAssaultBattery", \
              "Burglary","Larceny","MVT","MSO_Violent","All_Violent","Property"]
              
table_names = ['X_'+ctype+'_Pts_08_14' for ctype in crime_types]
csv_files = [file_path+ctype+'_08_14.csv' for ctype in crime_types]           

conn_str = 'IIT_USER/TSR1512@167.165.243.151:1521/dwhracdb.dwhrac.chicagopolice.local'
conn = cx_Oracle.connect(conn_str)

cursor = conn.cursor()
for table_name, csv_file in zip(table_names, csv_files):
    query_str = u'SELECT * FROM IIT_USER.' + table_name.upper() + ' ORDER BY DATEOCC ASC'
    cursor.execute(query_str)
    data_df = fetch_data_to_dataframe(cursor)
    # save as csv
    with open(csv_file, 'wb') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(data_df['header'])
        csv_writer.writerows(data_df['data'])
    
conn.close()

