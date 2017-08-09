#!/bin/bash
regex='([0-9]+ ([0-9]{4}-[0-9]{2}-[0-9]{2} ?){2})[[:<:]](True|False)[[:>:]] (([0-9]*\.[0-9]+ ?|[0-9]+ ?){7})(([0-9]{4}-[0-9]{2}-[0-9]{2} ?){4}) ([0-9]+ [0-9]+)'
structline=$(<../StructureData/struct_spec.txt)
param_group=$(echo $structline | sed -E "s/${regex}/\1/")
param_LTST=$(echo $structline | sed -E "s/${regex}/\4/")
param_traintest=$(echo $structline | sed -E "s/${regex}/\6/")
param_chunk=$(echo $structline | sed -E "s/${regex}/\8/")
param_targetcrimetypes=$(<../StructureData/target_crime_types.txt)

while IFS='' read -r cellline|| [[ -n "$cellline" ]]
do
    cellsize=($cellline)
    echo "Cell size: ${cellsize[0]} ${cellsize[1]}"
    loadfile_grid="../SharedData/SpatialData/grid_${cellsize[0]}_${cellsize[1]}/grid.pkl"
    loadpath_crime="../SharedData/CrimeData/"
    savepath="../SharedData/ModelData/grid_${cellsize[0]}_${cellsize[1]}/"
    mkdir -p $savepath

    python Baseline.py -i "${loadfile_grid} ${loadpath_crime}" -o ${savepath} -p "group=${param_group} traintest=${param_traintest} ltst=${param_LTST} targetcrimetypes=${param_targetcrimetypes} chunk=${param_chunk}"
done < ../Grid/cellsizes.txt
