#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
NOTE :
# one may need to set environment varibale for oracle clinet
# before importing cx_Oracle

# Create a .profile file with following 
# export ORACLE_HOME=/Users/xiaomuliu/Oracle/instantclient_12_1
# export DYLD_LIBRARY_PATH=$ORACLE_HOME
# export LD_LIBRARY_PATH=$ORACLE_HOME
#
# Then run
# $ source .profile
# $ echo $ORACLE_HOME
# $ echo $DYLD_LIBRARY_PATH
# $ echo $LD_LIBRARY_PATH

Created on Fri Sep 30 11:02:06 2016

@author: xiaomuliu
"""

import cx_Oracle
import json
import csv
from datetime import datetime

def fetch_data_to_dictlist(cursor):
    dict_list = [dict((cursor.description[i][0], value) \
         for i, value in enumerate(row)) for row in cursor.fetchall()]
    return dict_list

def fetch_data_to_dataframe(cursor):
    data = cursor.fetchall()
    header = [des[0] for des in cursor.description]
    return {'header':header,'data':data}

def serialize_datetime(dict_list, date_format='%Y-%m-%d'):
    """
    # NOTE: datetime.datetime object is not JSON serializable
    # One solution is to convert datetime.datetime objects to strings.
    """
    serial_dict_list = []
    for dict_obj in dict_list:
        for key, value in dict_obj.items():
            if isinstance(value, list):
                if isinstance(value[0], datetime):
                    # here assuming all elements in 'value' are of type datetime 
                    dict_obj[key] = [v.strftime(date_format) for v in value]
            else:                       
                if isinstance(value, datetime):
                    dict_obj[key] = value.strftime(date_format)
        serial_dict_list.append(dict_obj)        
    
    return serial_dict_list
    
    
if __name__=='__main__':   
    # Build connection string
    usr = "IIT_USER"
    pwd = "TSR1512"
    host = "167.165.243.151"
    port = "1521"
    service_name = "dwhracdb.dwhrac.chicagopolice.local"
    dsn = cx_Oracle.makedsn(host, port, service_name=service_name)                  
    conn = cx_Oracle.connect(usr, pwd, dsn)
    ## Equivalently,
    ## connection string format:'user/password@host:port/service'
    # conn_str = u'IIT_USER/TSR1512@167.165.243.151:1521/dwhracdb.dwhrac.chicagopolice.local'
    # conn = cx_Oracle.connect(conn_str)
    
    cursor = conn.cursor()
    query_str = u'SELECT * FROM IIT_USER.X_7DAY_COUNT WHERE RETRIEVAL_TIME > sysdate-1'
    cursor.execute(query_str)
    data_dictlist = fetch_data_to_dictlist(cursor)
    cursor.execute(query_str)
    data_df = fetch_data_to_dataframe(cursor)
    conn.close()
    
    # save as json
    data_dictlist = serialize_datetime(data_dictlist,'%m/%d/%Y %I:%M:%S %p')           
    json_file = 'test.json'
    with open(json_file, 'wb') as output_file:
        json.dump(data_dictlist, output_file)
    
    # save as csv
    csv_file = 'test.csv'
    with open(csv_file, 'wb') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(data_df['header'])
        csv_writer.writerows(data_df['data'])