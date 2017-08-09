#!/bin/bash

while IFS='' read -r cellline|| [[ -n "$cellline" ]]
do
    cellsize=($cellline)
    echo "Cell size: ${cellsize[0]} ${cellsize[1]}"
    loadfile_grid="../SharedData/SpatialData/grid_${cellsize[0]}_${cellsize[1]}/grid.pkl"
    loadpath_GIS="../SharedData/GISData/"
    loadpath_census="../SharedData/CensusData/"
    loadpath_POD="../SharedData/PODdata/"
    savepath="../SharedData/FeatureData/grid_${cellsize[0]}_${cellsize[1]}/"
    mkdir -p $savepath
    patchratio=$(<./patch_ratios.txt)
    python SpatialFeatureExtraction.py -i "${loadfile_grid} ${loadpath_GIS} ${loadpath_census} ${loadpath_POD}" -o ${savepath} -p "${patchratio}"
done < ../Grid/cellsizes.txt