#!/bin/bash
while IFS='' read -r exp_spec|| [[ -n "$exp_spec" ]]
do
    exp_spec=($exp_spec)
    echo "Target Crime: ${exp_spec[0]}; Cell size: ${exp_spec[1]} ${exp_spec[2]}"

    loadfile_featurerank="./Evaluation/grid_${exp_spec[1]}_${exp_spec[2]}/FeatureRank/${exp_spec[0]}_feature_ranking_dataframe1.pkl"

    savepath="../SharedData/FeatureData/grid_${exp_spec[1]}_${exp_spec[2]}/Ranking/"
    mkdir -p $savepath

    python RankVoting.py -i "targetcrime=${exp_spec[0]} featurerank=${loadfile_featurerank}" -o "${savepath}" -p "Nfeatures=${exp_spec[3]}"
done < ./target_crime_list.txt