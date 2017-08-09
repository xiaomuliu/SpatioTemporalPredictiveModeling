#!/bin/bash
param_repro=($(<../ModelSegmentation/repro_specs.txt))
while IFS='' read -r exp_spec|| [[ -n "$exp_spec" ]]
do
    exp_spec=($exp_spec)
    echo "Target Crime: ${exp_spec[0]}; Cell size: ${exp_spec[1]} ${exp_spec[2]}"

    loadfile_feature="../SharedData/FeatureData/grid_${exp_spec[1]}_${exp_spec[2]}/balanced_feature_target_dataframe.h5"

    savepath="./Evaluation/grid_${exp_spec[1]}_${exp_spec[2]}/"
    mkdir -p $savepath

    python BaggingFeatureSelection.py -i "targetcrime=${exp_spec[0]} feature=${loadfile_feature}" -o "${savepath}" -p "kfolds=${param_repro[0]} rseed=${param_repro[1]}"
done < ./target_crime_list.txt
