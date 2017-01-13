# -*- coding: utf-8 -*-
"""
===================================
Manipulating Crime Data
===================================
Created on Thu Aug 18 10:38:37 2016

@author: xiaomuliu
"""
import pandas as pd
from ImportCrimeData import FilterDataByCrimeType

filePath = "./CrimeData/Portal/"
fileName_load = "Crimes_2001_to_present.csv"
loadfile = filePath + fileName_load 
CrimeData = pd.read_csv(loadfile)

CrimeTypes = ["Homicide","CriminalSexualAssault","Robbery","AggravatedAssault","AggravatedBattery",\
                "Burglary","Larceny","MotorVehicleTheft","Arson"]
for crimetype in CrimeTypes:
    FilteredCrimeData = FilterDataByCrimeType(CrimeData,crimetype)
    fileName_save = crimetype.upper()+"_01_14.csv"
    FilteredCrimeData.to_csv(fileName_save,index=False)

