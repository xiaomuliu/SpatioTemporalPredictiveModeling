#!/bin/bash
while IFS='' read -r exp_spec|| [[ -n "$exp_spec" ]]
do
    exp_spec=($exp_spec)
    echo "Target crime type: ${exp_spec[0]}; Cell size: ${exp_spec[1]} ${exp_spec[2]}; Clustering Model: ${exp_spec[3]}"

    loadfile_grid="../SharedData/SpatialData/grid_${exp_spec[1]}_${exp_spec[2]}/grid.pkl"
    loadfile_crime="../SharedData/CrimeData/${exp_spec[0]}_08_14_grouped.pkl"
    loadfile_group="../SharedData/FeatureData/grid_${exp_spec[1]}_${exp_spec[2]}/train_test_groups.pkl"
    loadfile_feature="../SharedData/FeatureData/grid_${exp_spec[1]}_${exp_spec[2]}/${exp_spec[0]}_feature_label_dataframe.pkl'"
    loadfile_feature_b="../SharedData/FeatureData/grid_${exp_spec[1]}_${exp_spec[2]}/${exp_spec[0]}_balanced_feature_label_dataframe.pkl'"
    loadfile_baseline="../SharedData/ModelData/grid_${exp_spec[1]}_${exp_spec[2]}/baseline_dataframe.pkl"
    loadfile_cluster="../SharedData/SpatialData/grid_${exp_spec[1]}_${exp_spec[2]}/clusters_${exp_spec[3]}.pkl"
    loadfile_mask="../SharedData/SpatialData/grid_${exp_spec[1]}_${exp_spec[2]}/masks_${exp_spec[3]}.pkl"

    param_repro=($(<./repro_specs.txt))
    param_eval=$(<./eval_specs.txt)
    savepath="./Evaluation/grid_${exp_spec[1]}_${exp_spec[2]}/${exp_spec[0]}/${exp_spec[3]}/${exp_spec[4]}/"
    mkdir -p $savepath

    python ModelSegmentation.py -i "grid=${loadfile_grid} crime=${loadfile_crime} group=${loadfile_group} feature=${loadfile_feature} featureB=${loadfile_feature_b} baseline=${loadfile_baseline} cluster=${loadfile_cluster} mask=${loadfile_mask}" -o ${savepath} -p "targetcrime=${exp_spec[0]} model=${exp_spec[4]} kfolds=${param_repro[0]} rseed=${param_repro[1]} eval=${param_eval}" > "${savepath}/experiment_log.txt"
done < ./experiment_specs.txt
