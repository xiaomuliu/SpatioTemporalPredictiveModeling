# -*- coding: utf-8 -*-
"""
====================================
load crime csv files and save as python pickles
====================================

Created on Mon Aug 22 11:27:14 2016

@author: xiaomuliu
"""
import cPickle as pickle
from ImportCrimeData import ImportCrimeData

filePath = "/Users/xiaomuliu/CrimeProject/SpatioTemporalModeling/DataProcessing/CrimeData/csv_py/"
CrimeTypes = ["Homicide","CriminalSexualAssault","Robbery","AggravatedAssault","AggravatedBattery",\
              "Burglary","Larceny","MotorVehicleTheft","Arson","ViolentCrime"]
              
for crimetype in CrimeTypes:
    loadfile = filePath + crimetype.upper()+'_01_14.csv'
    CrimeData = ImportCrimeData(loadfile)
    savefile = filePath + crimetype.upper()+'_01_14.pkl'
    with open(savefile,'wb') as output:
        pickle.dump(CrimeData, output, pickle.HIGHEST_PROTOCOL)

# verify
pkl_file = open(filePath+'HOMICIDE_01_14.pkl', 'rb')
HomicideData = pickle.load(pkl_file)
pkl_file.close()
HomicideData.head()