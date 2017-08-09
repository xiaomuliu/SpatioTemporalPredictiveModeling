#!/bin/bash
param_repro=($(<./repro_specs.txt))

regex='([0-9]+ ([0-9]{4}-[0-9]{2}-[0-9]{2} ?){2})[[:<:]](True|False)[[:>:]] (([0-9]*\.[0-9]+ ?|[0-9]+ ?){7})(([0-9]{4}-[0-9]{2}-[0-9]{2} ?){4}) ([0-9]+ [0-9]+)'
structline=$(<../StructureData/struct_spec.txt)
param_chunk=$(echo $structline | sed -E "s/${regex}/\8/")

while IFS='' read -r exp_spec|| [[ -n "$exp_spec" ]]
do
exp_spec=($exp_spec)
target_crime=${exp_spec[0]}
cellsize=(${exp_spec[1]} ${exp_spec[2]})
cluster_model=${exp_spec[3]}
train_region=${exp_spec[4]}
train_region_num=${exp_spec[5]}
test_region=${exp_spec[6]}
test_region_num=${exp_spec[7]}
pred_model=${exp_spec[8]}

if [ $train_region_num == "NA" ]
then
train_region_num=""
fi
if [ $test_region_num == "NA" ]
then
test_region_num=""
fi

echo "Target crime type: ${exp_spec[0]}; Cell size: ${exp_spec[1]} ${exp_spec[2]}; Clustering Model: ${cluster_model}; Training examples: ${train_region} ${train_region_num}; Test examples: ${test_region} ${test_region_num}; Predictive model: ${pred_model}"

path_featurefile="../SharedData/FeatureData/grid_${cellsize[0]}_${cellsize[1]}/Tr12_13_Te14/"
path_modelfile="../SharedData/ModelData/grid_${cellsize[0]}_${cellsize[1]}/Tr12_13_Te14/"
path_spatialfile="../SharedData/SpatialData/grid_${cellsize[0]}_${cellsize[1]}/"

loadfile_group="${path_featurefile}train_test_groups.pkl"
loadfile_test=$( ls ${path_featurefile}feature_target_test_*_dataframe.h5)
if [ $cluster_model != "NA" ]
then
loadfile_cluster="${path_spatialfile}cluster/clusters_${cluster_model}.pkl"
else
loadfile_cluster="NA"
fi
if [ $train_region == "district" ] || [ $test_region == "district" ]
then
loadfile_district="${path_spatialfile}label_district.pkl"
else
loadfile_district="NA"
fi

if [ $train_region == "city" ]
then
loadfile_train="${path_featurefile}feature_target_train_dataframe.h5"
savepath="${path_modelfile}${target_crime}/city/${pred_model}/"
elif [ $train_region == "cluster" ]
then
loadfile_train=$( ls ${path_featurefile}feature_target_train_*_dataframe.h5)
savepath="${path_modelfile}${target_crime}/${cluster_model}/${pred_model}/"
elif [ $train_region == "district" ]
then
loadfile_train=$( ls ${path_featurefile}feature_target_train_*_dataframe.h5)
savepath="${path_modelfile}${target_crime}/district/${pred_model}/"
fi

mkdir -p $savepath

python FitPredModel.py -i "group=${loadfile_group} traindata=${loadfile_train} testdata=${loadfile_test} cluster=${loadfile_cluster} district=${loadfile_district}" -o ${savepath} -p "targetcrime=${target_crime} model=${pred_model} kfolds=${param_repro[0]} rseed=${param_repro[1]} trainregion=${train_region} trainregionNo=${train_region_num} testregion=${test_region} testregionNo=${test_region_num} chunksize=${param_chunk}" > "${savepath}experiment_log.txt"
done < ./experiment_specs_class.txt
