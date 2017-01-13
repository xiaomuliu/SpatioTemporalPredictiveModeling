#!/bin/bash
# Munge raw data and save them as pickle files

python MungeCrimeData.py -i ./CrimeData/DWH/ -o ../SharedData/CrimeData/
python Munge311CallsData.py -i ./OtherData/ -o ../SharedData/311Data/
python MungeWeatherData.py -i ./WeatherData/WeatherData_Daily_2001-01-01_2014-12-31.csv -o ../SharedData/WeatherData/Weather_01_14.pkl
python MungePODdata.py -i ./OtherData/PODs.csv -o ../SharedData/PODdata/PODs_05_14.pkl