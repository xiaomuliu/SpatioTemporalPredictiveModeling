#!/bin/bash

district_shp="../SharedData/GISData/cpd_districts/cpd_districts.shp"

while IFS='' read -r line|| [[ -n "$line" ]]
do
    cellsize=($line)
    echo "Cell size: ${cellsize[0]} ${cellsize[1]}"
    grid_pkl="../SharedData/SpatialData/grid_${cellsize[0]}_${cellsize[1]}/grid.pkl"
    savepath="../SharedData/SpatialData/grid_${cellsize[0]}_${cellsize[1]}/"

    python DistrictMask.py -i "${grid_pkl} ${district_shp}" -o "${savepath}" -p "${cellsize[0]} ${cellsize[1]}"
done < cellsizes.txt
