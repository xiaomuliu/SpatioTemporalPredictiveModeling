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

# The following two ways, however, do not work. It seems DYLD_LIBRARY_PATH and 
# LD_LIBRARY_PATH should be set in another way. Need to figure out how.
# Method 1:
# import os
# oracle_home = '/Users/xiaomuliu/Oracle/instantclient_12_1'
# os.environ['ORACLE_HOME'] = oracle_home
# os.environ['DYLD_LIBRARY_PATH'] = oracle_home
# os.environ['LD_LIBRARY_PATH'] = oracle_home

# Method 2:
# import os
# os.system('source /Users/xiaomuliu/.profile');

import cx_Oracle, getpass

# Get password
pswd = getpass.getpass() #TSR1512
    
# Build connection string
user = "IIT_USER"
host = "167.165.243.151"
port = "1521"
service_name = "dwhracdb.dwhrac.chicagopolice.local"
dsn = cx_Oracle.makedsn (host, port, service_name=service_name)
                  
# Connect to Oracle and test
con = cx_Oracle.connect (user, pswd, dsn)
if (con):
  print "Connection successful"
  print con.version
else:
  print "Connection not successful"

con.close()

