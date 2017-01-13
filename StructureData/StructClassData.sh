#!/bin/bash

# In python 2.x, package 'cPickle' will raise an error when saving objects larger than 4GB
# One can switch to python 3.x to do this job:
# source activate py35
# python xxx.py
# ....
# source deactivate
# However, the default pickling protocol used by default in Python 3 is incompatible with the protocol used by Python 2, which means loading/saving pickle files between these two verisons needs extra work
# In python 2, the default encoding is ASCII
# with open(picklefile, 'rb') as f:
#     d = pickle.load(f, encoding='latin1')

while IFS='' read -r cellline|| [[ -n "$cellline" ]]
do
    cellsize=($cellline)
    echo "Cell size: ${cellsize[0]} ${cellsize[1]}"
    loadfile_grid="../SharedData/SpatialData/grid_${cellsize[0]}_${cellsize[1]}/grid.pkl"
    loadpath_crime="../SharedData/CrimeData/"
    loadpath_weather="../SharedData/WeatherData/"
    loadpath_311="../SharedData/311data/"
    loadpath_POD="../SharedData/PODdata/"
    loadpath_spfeature="../SharedData/FeatureData/grid_${cellsize[0]}_${cellsize[1]}/"
    savepath="../SharedData/FeatureData/grid_${cellsize[0]}_${cellsize[1]}/"
    mkdir -p $savepath
    param_group=$(<./group_params.txt)
    param_traintest=$(<./date_range_train_test.txt)
    param_LTST=$(<./LT_ST_params.txt)
    python StructClassData.py -i "${loadfile_grid} ${loadpath_crime} ${loadpath_weather} ${loadpath_311} ${loadpath_POD} ${loadpath_spfeature}" -o ${savepath} -p "group=${param_group} traintest=${param_traintest} ltst=${param_LTST}"
done < ../Grid/cellsizes.txt
