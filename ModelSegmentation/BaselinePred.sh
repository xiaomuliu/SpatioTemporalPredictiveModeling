#!/bin/bash
while IFS='' read -r cellline|| [[ -n "$cellline" ]]
do
    cellsize=($cellline)
    echo "Cell size: ${cellsize[0]} ${cellsize[1]}"
    loadfile_grid="../SharedData/SpatialData/grid_${cellsize[0]}_${cellsize[1]}/grid.pkl"
    loadpath_crime="../SharedData/CrimeData/"
    savepath="../SharedData/ModelData/grid_${cellsize[0]}_${cellsize[1]}/"
    mkdir -p $savepath
    param_group=$(<../StructureData/group_params.txt)
    param_traintest=$(<../StructureData/date_range_train_test.txt)
    param_LTST=$(<../StructureData/LT_ST_params.txt)
    python Baseline.py -i "${loadfile_grid} ${loadpath_crime}" -o ${savepath} -p "group=${param_group} traintest=${param_traintest} ltst=${param_LTST}"
done < ../Grid/cellsizes.txt
