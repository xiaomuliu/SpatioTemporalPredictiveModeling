#!/bin/bash
while IFS='' read -r line|| [[ -n "$line" ]]
do
cellsize=($line)
echo "Cell size: ${cellsize[0]} ${cellsize[1]}"
path="../SharedData/SpatialData/grid_${cellsize[0]}_${cellsize[1]}/"
python Graph.py -i "${path}grid.pkl" -o $path
done < cellsizes.txt
