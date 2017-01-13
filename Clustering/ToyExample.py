#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
=================================
Toy example (for verifying GMM_HMRF)
=================================
Created on Thu Oct 13 15:51:11 2016

@author: xiaomuliu
"""
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import Graph
import networkx as nx
from GMM_HMRF import GMM_HMRF_EM
from MRF_GMM import GMM_MRF
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture 
import time

im = misc.face()
# crop a sub region
M, N, P = im.shape
im = im[int(0.65*M):int(0.8*M),int(0.4*N):int(0.55*N),:]

plt.imshow(im)
plt.show()

# flatten image
Nx, Ny, P = im.shape

im_f = np.zeros((Nx*Ny,P))
for p in xrange(P):
    im_f[:,p] = im[:,:,p].flatten(order='C')

plt.imshow(im_f[:,0].reshape((Nx,Ny),order='C'))
    
#adj_idx_arr_list, adj_idx_ravel_list, adj_mask_idx_ravel_list = Graph.grd_to_adj_list(Nx, Ny, order='C')

start = time.time()
graph = nx.grid_2d_graph(Nx,Ny)        
adj_idx_arr_list = nx.to_dict_of_lists(graph)
adj_idx_ravel_list = Graph.arr_idx_to_flat_idx(adj_idx_arr_list, Nx, Ny, order='C')
end = time.time()
print(end-start)

zscore_scaler = preprocessing.StandardScaler()
X_s = zscore_scaler.fit_transform(im_f)
        
r_seed = 1234 
Ncomponents = 4
beta = 0.1
cov_type = 'full'
GMM_init = GaussianMixture(n_components=Ncomponents, covariance_type=cov_type,
                           init_params='kmeans', max_iter=50, random_state=r_seed).fit(X_s)

# Inital GMM clustering
y_init= GMM_init.predict(X_s) 
plt.matshow(y_init.reshape((Nx,Ny),order='C'), interpolation='nearest', origin='upper')

 
#start = time.time()
#mrf1 = GMM_HMRF_EM(X_s, beta, adj_idx_ravel_list, GMM_init, EM_max_iter=50, ICM_max_iter=20, soft=True)
#end = time.time()
#print(end-start)
#print('Convergence: %d' % mrf1['convergence'])
#plt.matshow(mrf1['label'].reshape((Nx,Ny),order='C'), interpolation='nearest', origin='upper')


start = time.time()
mrf2 = GMM_HMRF_EM(X_s, beta, adj_idx_ravel_list, GMM_init, EM_max_iter=50, ICM_max_iter=20, soft=False)
end = time.time()
print(end-start)
print('Convergence: %d' % mrf2['convergence'])
plt.matshow(mrf2['label'].reshape((Nx,Ny),order='C'), interpolation='nearest', origin='upper')

start = time.time()
mrf3 = GMM_MRF(X_s, beta, adj_idx_ravel_list, GMM_init, EM_max_iter=50, ICM_max_iter=20)
end = time.time()
print(end-start)
print('Convergence: %d' % mrf3['convergence'])
plt.matshow(mrf3['label'].reshape((Nx,Ny),order='C'), interpolation='nearest', origin='upper')