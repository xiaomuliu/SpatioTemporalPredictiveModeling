#!/bin/bash
while IFS='' read -r line|| [[ -n "$line" ]]
do
param=($line)
echo "Group size: ${param[0]} Date range: ${param[1]} - ${param[2]}"
path="../SharedData/"
echo "Grouping crime data ..."
python GroupCrimeData.py -i "${path}CrimeData/" -o "${path}CrimeData/" -p "${param[0]} ${param[1]} ${param[2]}"
echo "Grouping Weather data ..."
python GroupWeatherData.py -i "${path}WeatherData/" -o "${path}WeatherData/" -p "${param[0]} ${param[1]} ${param[2]}"
echo "Grouping 311 data ..."
python Group311Data.py -i "${path}311Data/" -o "${path}311Data/" -p "${param[0]} ${param[1]} ${param[2]}"
echo "Grouping POD data ..."
python GroupPODData.py -i "${path}PODdata/" -o "${path}PODdata/" -p "${param[0]} ${param[1]} ${param[2]}"
done < group_params.txt
