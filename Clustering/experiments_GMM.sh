#!/bin/bash
# read the parameter (number of Gaussian components) from 'params_GMM.txt'. Run the experiments for different grid cell sizes. Save the results to 'GMM_exp_results.txt'
while IFS='' read -r cellline|| [[ -n "$cellline" ]]
do
    cellsize=($cellline)
    echo "Cell size: ${cellsize[0]} ${cellsize[1]}"
    loadfile_grid="../SharedData/SpatialData/grid_${cellsize[0]}_${cellsize[1]}/grid.pkl"
    loadfile_feature="./FeatureData/grid_${cellsize[0]}_${cellsize[1]}/feature_dataframe.pkl"
    while IFS='' read -r Ncomp|| [[ -n "$Ncomp" ]]
    do
        echo "Number of Components: ${Ncomp}"
        savepath_fig="./Figures/grid_${cellsize[0]}_${cellsize[1]}/GMM/Ncomp_${Ncomp}/"
        mkdir -p $savepath_fig
        python GMM.py -i "${loadfile_grid} ${loadfile_feature}" -o ${savepath_fig} -p ${Ncomp}
    done < params_GMM.txt
done < ../Grid/cellsizes.txt > GMM_exp_results.txt