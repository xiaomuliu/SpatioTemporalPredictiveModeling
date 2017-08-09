#!/bin/bash
loadpath_crime="../SharedData/CrimeData/"
loadpath_weather="../SharedData/WeatherData/"
loadpath_311="../SharedData/311data/"
loadpath_POD="../SharedData/PODdata/"

regex='([0-9]+ ([0-9]{4}-[0-9]{2}-[0-9]{2} ?){2})[[:<:]](True|False)[[:>:]] (([0-9]*\.[0-9]+ ?|[0-9]+ ?){4}) (([0-9]+ ?){1}) (([0-9]{4}-[0-9]{2}-[0-9]{2} ?){4})'
structline=$(<./seq_struct_spec.txt)
param_group=$(echo $structline | sed -E "s/${regex}/\1/")
param_label=$(echo $structline | sed -E "s/${regex}/\3/")
param_LT=$(echo $structline | sed -E "s/${regex}/\4/")
param_seqlen=$(echo $structline | sed -E "s/${regex}/\6/")
param_traintest=$(echo $structline | sed -E "s/${regex}/\8/")
#sed backreference only supports 0-9
regex2='([0-9]+ [0-9]+)'
chunkline=$(<./chunk_sizes.txt)
param_chunk=$(echo $chunkline | sed -E "s/${regex2}/\1/")
param_featurecrimetypes=$(<./feature_crime_types.txt)
param_targetcrimetypes=$(<./target_crime_types.txt)
while IFS='' read -r cellline|| [[ -n "$cellline" ]]
do
cellsize=($cellline)
echo "Cell size: ${cellsize[0]} ${cellsize[1]}"
loadfile_grid="../SharedData/SpatialData/grid_${cellsize[0]}_${cellsize[1]}/grid.pkl"
loadpath_spfeature="../SharedData/FeatureData/grid_${cellsize[0]}_${cellsize[1]}/"
savepath="../SharedData/FeatureData/grid_${cellsize[0]}_${cellsize[1]}/Seq/Tr12_13_Te14/"
mkdir -p $savepath
python StructSeqPredModelData.py -i "${loadfile_grid} ${loadpath_crime} ${loadpath_weather} ${loadpath_311} ${loadpath_POD} ${loadpath_spfeature}" -o ${savepath} -p "group=${param_group} traintest=${param_traintest} longterm=${param_LT} seqlen=${param_seqlen} label=${param_label} featurecrimetypes=${param_featurecrimetypes} targetcrimetypes=${param_targetcrimetypes} chunk=${param_chunk}"
done < ../Grid/cellsizes.txt
