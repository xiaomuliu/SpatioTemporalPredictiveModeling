#!/bin/bash
while IFS='' read -r line|| [[ -n "$line" ]]
do
   cellsize=($line)
   echo "Cell size: ${cellsize[0]} ${cellsize[1]}"
   savepath="../SharedData/SpatialData/grid_${cellsize[0]}_${cellsize[1]}/"
   mkdir -p $savepath
   python CreateGrid.py -i ../SharedData/GISData/City_Boundary/City_Boundary.shp -o $savepath -p "${cellsize[0]} ${cellsize[1]}"
done < cellsizes.txt