#!/bin/bash
param_eval=$(<./eval_specs.txt)
declare -a loadfile_predscore
declare -a loadfile_samplemask
declare -a model_names
counter=0
while IFS='' read -r exp_spec|| [[ -n "$exp_spec" ]]
do
	exp_spec=($exp_spec)
	# The first line specifies target crime type, cell size and evaluation region
	if [ $counter -eq 0 ]
	then
		target_crime=${exp_spec[0]}
		cellsize=(${exp_spec[1]} ${exp_spec[2]})
		eval_region=${exp_spec[3]}
		loadfile_cluster="NA"
        loadfile_clustermask="NA"
		loadfile_district="NA"
		loadfile_districtmask="NA"

        path_spatialfile="../SharedData/SpatialData/grid_${cellsize[0]}_${cellsize[1]}/"
        path_modelfile="../SharedData/ModelData/grid_${cellsize[0]}_${cellsize[1]}/Seq/Tr12_13_Te14/${target_crime}/"

		if [ $eval_region == "city" ]
		then
		eval_region_num="NA"
		elif [ $eval_region == "cluster" ]
		then 
		cluster_model=${exp_spec[4]}

        loadfile_cluster="${path_spatialfile}cluster/clusters_${cluster_model}.pkl"
        loadfile_clustermask="${path_spatialfile}cluster/masks_${cluster_model}.pkl"

		eval_region_num=${exp_spec[5]}
		elif [ $eval_region == 'district' ]
		loadfile_district="${path_spatialfile}label_district.pkl"
        loadfile_districtmask="${path_spatialfile}masks_district.pkl"
		then
		eval_region_num=${exp_spec[4]}
		fi
		
		loadfile_grid="${path_spatialfile}grid.pkl"
		loadfile_crime="../SharedData/CrimeData/${target_crime}_08_14_grouped.pkl"
        loadfile_group="../SharedData/FeatureData/grid_${cellsize[0]}_${cellsize[1]}/Seq/Tr12_13_Te14/train_test_groups.pkl"

		loadfile_baseline="../SharedData/ModelData/grid_${cellsize[0]}_${cellsize[1]}/baseline_dataframe.pkl"		
		((counter++))
	continue
	fi

	cluster_model=${exp_spec[0]}
	train_region=${exp_spec[1]}
	train_region_num=${exp_spec[2]}
	test_region=${exp_spec[3]}
	test_region_num=${exp_spec[4]}
	pred_model=${exp_spec[5]}
	((counter++))

	if [ $train_region_num == "NA" ]
	then
		train_region_num=""
	fi
	if [ $test_region_num == "NA" ]
	then
		test_region_num=""
	fi

	echo "Target crime type: ${target_crime}; Cell size: ${cellsize[0]} ${cellsize[1]}; Clustering Model: ${cluster_model}; Training examples: ${train_region} ${train_region_num}; Test examples: ${test_region} ${test_region_num}; Predictive model: ${pred_model}"

	if [ $train_region == "city" ]
	then
		predscore="${path_modelfile}city/${pred_model}/PredScore_city_city.csv"
		samplemask="${path_modelfile}city/${pred_model}/SampleMask_city_city.pkl"
	elif [ $train_region == "cluster" ]
	then
		predscore="${path_modelfile}${cluster_model}/${pred_model}/PredScore_${train_region}${train_region_num}_${test_region}${test_region_num}.csv"
		samplemask="${path_modelfile}${cluster_model}/${pred_model}/SampleMask_${train_region}${train_region_num}_${test_region}${test_region_num}.pkl"
	elif [ $train_region == "district" ]
	then
		predscore="${path_modelfile}district/${pred_model}/PredScore_${train_region}${train_region_num}_${test_region}${test_region_num}.csv"
		samplemask="${path_modelfile}district/${pred_model}/SampleMask_${train_region}${train_region_num}_${test_region}${test_region_num}.pkl"
	fi

	loadfile_predscore=("${loadfile_predscore[@]}" "${predscore};") # append a predscore file to the array $loadfile_predscore. Use ';' to seperate elements
	loadfile_samplemask=("${loadfile_samplemask[@]}" "${samplemask};") # append a samplemask file to the array $loadfile_samplemask. Use ';' to seperate elements
	model_names=("${model_names[@]}" "${pred_model} clustering:${cluster_model} training:${train_region}${train_region_num};")
	
done < ./eval_experiment_specs_class.txt

savepath="./Evaluation/grid_${cellsize[0]}_${cellsize[1]}/${target_crime}/Seq/Tr12_13_Te14/MultiComparison/${eval_region}${eval_region_num}/"
mkdir -p $savepath

python EvalSeqPredModel.py -i "grid=${loadfile_grid} crime=${loadfile_crime} group=${loadfile_group} baseline=${loadfile_baseline} predscore=${loadfile_predscore[@]} samplemask=${loadfile_samplemask[@]} cluster=${loadfile_cluster} clustermask=${loadfile_clustermask} district=${loadfile_district} districtmask=${loadfile_districtmask}" -o ${savepath} -p "targetcrime=${target_crime} evalregion=${eval_region} evalregionNo=${eval_region_num} model=${model_names[@]} evalspec=${param_eval}"


