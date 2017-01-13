#!/bin/bash
# read the parameters (number of Gaussian components and Dirichlet prior) from 'params_VBGMM.txt'. Run the experiments for different grid cell sizes.
while IFS='' read -r cellline|| [[ -n "$cellline" ]]
do
    cellsize=($cellline)
    echo "Cell size: ${cellsize[0]} ${cellsize[1]}"
    loadfile_grid="../SharedData/SpatialData/grid_${cellsize[0]}_${cellsize[1]}/grid.pkl"
    loadfile_feature="./FeatureData/grid_${cellsize[0]}_${cellsize[1]}/feature_dataframe.pkl"
    while IFS='' read -r paramline|| [[ -n "$paramline" ]]
    do
        param=($paramline)
        echo "Number of Components: ${param[0]}, Dirichlet Prior Parameter: ${param[1]}"
        savepath_fig="./Figures/grid_${cellsize[0]}_${cellsize[1]}/VBGMM/Ncomp_${param[0]}/"
        mkdir -p $savepath_fig
        python VBGMM.py -i "${loadfile_grid} ${loadfile_feature}" -o ${savepath_fig} -p "${param[0]} ${param[1]}"
    done < params_VBGMM.txt
done < ../Grid/cellsizes.txt
