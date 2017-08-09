#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:33:46 2017

For modifying model (training) specifications and tuning parameters in the experiments

@author: xiaomuliu
"""
import numpy as np
from sklearn import linear_model, ensemble, neural_network
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.wrappers.scikit_learn import KerasClassifier

def config_LSTM_arch(layers, feature_dim, timesteps, optimizer='rmsprop'):
    # expected input data shape: (batch_size, timesteps, feature_dim)

    # layers: a list/tuple where each element specifies the number of weights in the layer (excluding the output layer)
    model = Sequential()
    # The input layer
    if hasattr(layers,'__len__'):
        model.add(LSTM(layers[0], return_sequences=True,
                   input_shape=(timesteps, feature_dim))) #In input_shape, the batch dimension is not included
#    model.add(LSTM(layers[0], return_sequences=True,
#                   batch_input_shape=(batch_size, timesteps, feature_dim)))
        if len(layers)>1:
            #multiple LSTM layers
            for num_w in layers[1:-1]:
                model.add(LSTM(num_w, return_sequences=True))  # returns a sequence of vectors
            
            # The last LSTM layer returns a single vector instead of a sequence of vectors    
            model.add(LSTM(layers[-1]))
    else:
        # singla layer
        model.add(LSTM(layers,input_shape=(timesteps, feature_dim)))

    # The output layer
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model


def get_model_params(model_name, rand_seed=1234, X=None):
    if model_name=='logit_l2':
        fixed_params = dict(penalty='l2', solver='liblinear', n_jobs=1)
        model = linear_model.LogisticRegression(**fixed_params)    
        #tuning_params = dict(C=np.logspace(-3, 2, 11))
        tuning_params = dict(C=np.logspace(-2, 1, 4))
    elif model_name=='logit_l1':
        fixed_params = dict(penalty='l1', solver='liblinear', n_jobs=1)
        model = linear_model.LogisticRegression(**fixed_params)    
        #tuning_params = dict(C=np.logspace(-3, 2, 11))
        tuning_params = dict(C=np.logspace(-2, 1, 4))
    elif model_name=='RF':
        fixed_params = dict(min_samples_split = 20, bootstrap=True, random_state=rand_seed, n_jobs=-1)
        model = ensemble.RandomForestClassifier(**fixed_params)
        #tuning_params = dict(n_estimators=[500,1000,2000],max_features=['auto',0.3,0.6], min_samples_split = [20,50,100]) # 'auto' max_features=sqrt(n_features) 
        tuning_params = dict(n_estimators=[500,1000,1500],max_features=['auto',0.3]) 
    elif model_name=='GBM':
        fixed_params = dict(learning_rate=0.05, subsample=0.9, max_features=0.5, min_samples_leaf=1, random_state=rand_seed)
        model = ensemble.GradientBoostingClassifier(**fixed_params)
        #tuning_params = dict(n_estimators=np.arange(100,250,50),max_depth=[3,4,5],min_samples_split=np.arange(50,200,50))
#        tuning_params = dict(n_estimators=np.arange(150,250,50),max_depth=[4,5,6])
        tuning_params = dict(n_estimators=np.arange(50,300,50),max_depth=[3,4,5,6])
    elif model_name=='NN_relu':
        # ReLu neural network using 'adam' solvor
        fixed_params = dict(activation='relu', beta_1=0.9, beta_2=0.999, solver='adam', learning_rate_init=0.05, \
                            learning_rate='constant', random_state=rand_seed)
        model = neural_network.MLPClassifier(**fixed_params)
        #tuning_params = dict(hidden_layer_sizes=[(100,10),(50,5),(30,3)], alpha=np.logspace(-2,2,5), batch_size=np.arange(200,1200,400))
        tuning_params = dict(hidden_layer_sizes=[(100,10),(50,5),(30,3)], alpha=np.logspace(-1,1,3), batch_size=np.arange(200,1200,400)) 
    elif model_name=='NN_sigmoid':
        # ReLu neural network using 'adam' solvor
        fixed_params = dict(activation='logistic', beta_1=0.9, beta_2=0.999, solver='adam', learning_rate_init=0.05, \
                            learning_rate='constant', random_state=rand_seed)
        model = neural_network.MLPClassifier(**fixed_params)
        tuning_params = dict(hidden_layer_sizes=[(100,10),(50,5),(30,3)], alpha=np.logspace(-2,2,5), batch_size=np.arange(200,1200,400))      
    elif model_name=='LSTM':
        feature_dim, timesteps = X.shape[2], X.shape[1]
        fixed_params = dict(build_fn=config_LSTM_arch, verbose=0, feature_dim=feature_dim, timesteps=timesteps, 
                            optimizer='rmsprop', epochs=150, batch_size=512)
        model = KerasClassifier(**fixed_params)   
        #tuning_params = dict(layers=[64, 32, 16, 8, (64,64), (64,32), (64,16),(32,32),(32,16),(16,16),(16,8),(8,8)])
        tuning_params = dict(layers=[64,32,16,(128,64),(64,16,16),(64,32,16,8)])
        
    return dict(model=model,tuning_params=tuning_params)    
    
def get_cv_obj(kfolds=5, rand_seed=1234):    
    return StratifiedKFold(n_splits=kfolds, shuffle=False, random_state=rand_seed)    