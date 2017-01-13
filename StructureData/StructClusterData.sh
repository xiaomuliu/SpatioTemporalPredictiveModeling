#!/bin/bash
while IFS='' read -r cellline|| [[ -n "$cellline" ]]
do
    cellsize=($cellline)
    echo "Cell size: ${cellsize[0]} ${cellsize[1]}"
    loadfile_grid="../SharedData/SpatialData/grid_${cellsize[0]}_${cellsize[1]}/grid.pkl"
    loadpath_crime="../SharedData/CrimeData/"
    savepath="../Clustering/FeatureData/grid_${cellsize[0]}_${cellsize[1]}/"
    mkdir -p $savepath
    while IFS='' read -r dateline|| [[ -n "$dateline" ]]
    do
        daterange=($dateline)
        echo "Processing feature data of date range: ${daterange[0]} - ${daterange[1]} ..."
        python StructClusterData.py -i "${loadfile_grid} ${loadpath_crime}" -o ${savepath} -p "${cellsize[0]} ${cellsize[1]} ${daterange[0]} ${daterange[1]}"
    done < date_range_clustering.txt
done < ../Grid/cellsizes.txt