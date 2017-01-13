#!/bin/bash
# read the parameters (number of Gaussian components, smoothing prior parameter, Dirichlet prior parameter) from 'params_HMRF.txt'. Run the experiments for different grid cell sizes. Save the results to 'HMRF_exp_results.txt'
while IFS='' read -r cellline|| [[ -n "$cellline" ]]
do
    cellsize=($cellline)
    echo "Cell size: ${cellsize[0]} ${cellsize[1]}"
    loadfile_grid="../SharedData/SpatialData/grid_${cellsize[0]}_${cellsize[1]}/grid.pkl"
    loadfile_graph="../SharedData/SpatialData/grid_${cellsize[0]}_${cellsize[1]}/graph.pkl"
    loadfile_feature="./FeatureData/grid_${cellsize[0]}_${cellsize[1]}/feature_dataframe.pkl"
    while IFS='' read -r paramline|| [[ -n "$paramline" ]]
    do
        param=($paramline)
        echo "Number of Components: ${param[0]}, Gibbs Prior Parameter: ${param[1]} Dirichlet Prior Parameter: ${param[2]}"
        savepath_fig="./Figures/grid_${cellsize[0]}_${cellsize[1]}/GMM_HMRF/Ncomp_${param[0]}/beta_${param[1]}/"
        mkdir -p $savepath_fig
        savepath_cluster="../SharedData/SpatialData/grid_${cellsize[0]}_${cellsize[1]}/"
        python GMM_HMRF.py -i "${loadfile_grid} ${loadfile_graph} ${loadfile_feature}" -o "${savepath_fig} ${savepath_cluster}" -p "${param[0]} ${param[1]} ${param[2]}"
    done < params_HMRF.txt
done < ../Grid/cellsizes.txt > HMRF_exp_results.txt