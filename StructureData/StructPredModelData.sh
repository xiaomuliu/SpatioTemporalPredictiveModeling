#!/bin/bash
loadpath_crime="../SharedData/CrimeData/"
loadpath_weather="../SharedData/WeatherData/"
loadpath_311="../SharedData/311data/"
loadpath_POD="../SharedData/PODdata/"

regex='([0-9]+ ([0-9]{4}-[0-9]{2}-[0-9]{2} ?){2})[[:<:]](True|False)[[:>:]] (([0-9]*\.[0-9]+ ?|[0-9]+ ?){7})(([0-9]{4}-[0-9]{2}-[0-9]{2} ?){4}) ([0-9]+ [0-9]+)'
structline=$(<./struct_spec.txt)
param_group=$(echo $structline | sed -E "s/${regex}/\1/")
param_label=$(echo $structline | sed -E "s/${regex}/\3/")
param_LTST=$(echo $structline | sed -E "s/${regex}/\4/")
param_traintest=$(echo $structline | sed -E "s/${regex}/\6/")
param_chunk=$(echo $structline | sed -E "s/${regex}/\8/")
param_featurecrimetypes=$(<./feature_crime_types.txt)
param_targetcrimetypes=$(<./target_crime_types.txt)

while IFS='' read -r cellline|| [[ -n "$cellline" ]]
do
    cellsize=($cellline)
    echo "Cell size: ${cellsize[0]} ${cellsize[1]}"
    loadfile_grid="../SharedData/SpatialData/grid_${cellsize[0]}_${cellsize[1]}/grid.pkl"
    loadpath_spfeature="../SharedData/FeatureData/grid_${cellsize[0]}_${cellsize[1]}/"
    savepath="../SharedData/FeatureData/grid_${cellsize[0]}_${cellsize[1]}/Tr12_13_Te14/"
    mkdir -p $savepath
    python StructPredModelData.py -i "${loadfile_grid} ${loadpath_crime} ${loadpath_weather} ${loadpath_311} ${loadpath_POD} ${loadpath_spfeature}" -o ${savepath} -p "group=${param_group} traintest=${param_traintest} ltst=${param_LTST} label=${param_label} featurecrimetypes=${param_featurecrimetypes} targetcrimetypes=${param_targetcrimetypes} chunk=${param_chunk}"
done < ../Grid/cellsizes.txt