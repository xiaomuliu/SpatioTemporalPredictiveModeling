#!/bin/bash

# `sed' uses the POSIX basic regular expression syntax.  According to the standard, the meaning of some escape sequences is undefined in this syntax;  notable in the case of `sed' are `\|', `\+', `\?', `\`', `\'', `\<', `\>', `\b', `\B', `\w', and `\W'

while IFS='' read -r line|| [[ -n "$line" ]]
do
    cellsize=($line)
    echo "Cell size: ${cellsize[0]} ${cellsize[1]}"
    path="../SharedData/SpatialData/grid_${cellsize[0]}_${cellsize[1]}/cluster/"

    for file in $( ls ${path}clusters_*.pkl )
    do
        cluster_file=$( echo $file | sed -E 's/.*(clusters_[a-zA-Z0-9_]+\.pkl)/\1/' )
        echo "Processing $cluster_file ..."
        model_name=$( echo $cluster_file | sed -E 's/^clusters_([a-zA-Z0-9_]+)\.pkl$/\1/' )      
        python ClusterMask.py -i ${path}${cluster_file} -o ${path}masks_${model_name}.pkl -p "${cellsize[0]} ${cellsize[1]}"
    done

done < cellsizes.txt