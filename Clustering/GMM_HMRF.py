#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 15:52:51 2016

@author: xiaomuliu
"""

import numpy as np
from scipy.stats import multivariate_normal
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture 
from sklearn.mixture import BayesianGaussianMixture 
import cPickle as pickle
import VisualizeCluster as vc
import matplotlib.pyplot as plt


def get_gaussian_stats(GMMobj):
    """
    Return statistics of (Bayesian) Gaussians/mixture of Gaussians including:
    the number of components, mean, covariance and precision of each components.
    If GMMobj is an instance of skleanr.mixture.GaussianMixture,
    make covariance/precision matrix of shape (n_components, n_features, n_features)
    in GMMobj, the covariance/precision if of shpae
    (n_components,)                        if 'spherical',
    (n_features, n_features)               if 'tied',
    (n_components, n_features)             if 'diag',
    (n_components, n_features, n_features) if 'full' 
    """
    if isinstance(GMMobj,GaussianMixture) or isinstance(GMMobj,BayesianGaussianMixture):
        K = GMMobj.n_components
        weights = GMMobj.weights_
        means = GMMobj.means_
        P = means.shape[1]
        Covs = GMMobj.covariances_  
        cov_type = GMMobj.get_params()['covariance_type']              
        if cov_type=='spherical':
            Covs = np.array([var*np.eye(P) for var in Covs])
        elif cov_type=='tied':
            Covs = np.tile(Covs,(K,1)).reshape((K,P,P))
        elif cov_type=='diag':
            Covs = np.array([np.diag(var) for var in Covs])
        elif cov_type=='full':
            Covs = Covs
    else:
        # a dict containing means, covariances and precisions of a sequence of single component of Gaussians
        weights = GMMobj['weights_']
        means = GMMobj['means_']
        Covs = GMMobj['covariances_'] 
        K = len(weights)

    return dict(n_components=K, weights=weights, means=means, covariances=Covs)

def pdet(C, eps=1e-12):
    """
    Calculate pseudo-determinant of a matrix
    """    
    eigs = np.linalg.eig(C)
    C_det = np.product(eigs[eigs>eps])
    return C_det

def to_nonsingular(C, eps=1e-6):
    """
    Handle singular matrix (add a small positive value to its diagonal)
    """
    try:
        if C.shape[0]!=C.shape[1]:
            raise ValueError
    except ValueError:
        print "Matrix must be square" 
        
    if np.linalg.cond(C) < 1/sys.float_info.epsilon:
        C = C + np.eye(C.shape[0])*eps
    #if np.linalg.matrix_rank(C) != C.shape[0]:
    #    C = C + np.eye(C.shape[0])*ep
    return C   
    
def multinormal_pdf(X, mean, cov, log=False, pseudo_inv=False):
    """
    Return the probability of multivariate normal for each sample in X
    X is of shape (n_samples, n_variables)
    """
    N,P = X.shape
    
    precision = np.linalg.pinv(cov) if pseudo_inv else np.linalg.inv(cov)
    det_cov = pdet(cov) if pseudo_inv else np.linalg.det(cov)
        
    logpdf = -0.5*(np.diag(np.dot(np.dot((X- np.tile(mean,(N,1))), precision),(X- np.tile(mean,(N,1))).T)) \
                   + np.log(det_cov) + P*np.log(2*np.pi))

    return logpdf if log else np.exp(logpdf)     
    

def clique_protential(Y, adj_list, beta):
    """
    Calculate clique protential function using Potts model
    """
    N = len(Y)
    K = len(np.unique(Y))    
    U = np.zeros((N,K))
    for k in xrange(K):
        for i in xrange(N):
            U[i,k] = np.sum(Y[adj_list[i]] != k)
      
    U = beta * U
    return U

    
def Gibbs_prior_pdf(Y, adj_list, beta):
    """
    Suppose data is of shape (n_samples,) and has K unique labels,
    the returned pdf is of shape (n_samples, K), where each row is
    the probabilies of different labels.
    """
    U = clique_protential(Y, adj_list, beta)
    prob = np.exp(-U)
    
    prob = np.apply_along_axis(lambda x: x/float(np.sum(x)), axis=1, arr=prob) # row normalization
    return prob     
 
    
def label_posterior(X, Y, GMMobj, adj_list, beta, allow_singular=True):
    """
    Calculate membership posterior for each samples in X over all labels in Y,
    where prior is the Gibbs distribution determined by adjacency list and parameter beta
    The returned posterior is of shape (n_samples, n_labels)
    """
    G_stats = get_gaussian_stats(GMMobj)
    K = G_stats['n_components']
    means = G_stats['means']
    Covs = G_stats['covariances']
    N = X.shape[0]

    likelihood = np.zeros((N,K))
    prior = np.zeros((N,K))
    for k in xrange(K):
        likelihood[:,k] = multivariate_normal.pdf(X,mean=means[k,:],cov=Covs[k,:,:],allow_singular=allow_singular)
        #likelihood[:,k] = multinormal_pdf(X,mean=means[k,:],cov=Covs[k,:,:],log=False,pseudo_inv=allow_singular) 
         
    prior = Gibbs_prior_pdf(Y,adj_list,beta)
    posterior = likelihood * prior 
    posterior = np.apply_along_axis(lambda x: x/float(np.sum(x)), axis=1, arr=posterior) # row normalization 
    
    posterior = np.nan_to_num(posterior)
    return posterior

def label_indicator(X,Y):
    """
    (Hard) membership indicator for each sample
    The returned matrix is of shape (n_samples, n_labels)
    This is the h ard-clustering version in contrast to function 'label_posterior'
    """
    N = X.shape[0]
    K = len(np.unique(Y))
    membership = np.zeros((N,K))
    for k in xrange(K):
        membership[Y==k,k]=1
    return membership

    
def MLE_multinormal(X, weights=1):
    """
    Calculate (weighted) maximum likelihood estimation of multivariate Gaussian distribution
    X is of shape (n_samples, n_features)
    """
    N, P = X.shape
    weights = weights*np.ones(N) # ensure weights are of shape (n_sample,)
    
    w_hat = np.nanmean(weights)
    mean_hat = np.average(X,axis=0,weights=weights) 
    C = np.tile(np.sqrt(weights)[:,np.newaxis],(1,P)) * (X-np.tile(mean_hat,(N,1)))
    cov_hat = 1/float(np.nansum(weights)) * np.dot(C.T, C)
    
    return dict(weight=w_hat, mean=mean_hat, cov=cov_hat)
    
def MLE_GMM_stats(X, label_pos, nonsingular_cov=True):
    """
    Return a Gaussian-mixture-like dict which has items: 
    weights_: (n_components,) the weights of each components
    means_: (n_components,n_features),
    covariances_ : (n_components, n_features, n_features)
    """
    P = X.shape[1]
    K = label_pos.shape[1]
    weights = np.zeros(K)
    means = np.zeros((K,P))
    covariances = np.zeros((K,P,P))
    
    for k in xrange(K):
        MLE_X = MLE_multinormal(X, label_pos[:,k])
        weights[k] = MLE_X['weight']
        means[k,:] = MLE_X['mean']
        covariances[k,:,:] = MLE_X['cov']
    
    if nonsingular_cov:
        #This is not necessary if 'allow_singular' is set to True in 
        #multivariate_normal.pdf/logpdf (using pseudo-inverse)
        covariances = np.array([to_nonsingular(c) for c in covariances])
               
    return dict(weights_=weights, means_=means, covariances_=covariances)    

     
def MRF_MAP_ICM(X, Y_init, GMMobj, beta, adj_list, ICM_max_iter=20, rel_tol=1e-8, allow_singular=True):
    """
    The ICM algorithm for MAP estimation of MRFs
    Input:
    X: data of dimension NxP. (N: number of examples; P: number of variables)
    Y_init: initial labels
    GMMobj: Gaussian mixture model object/a dict containing means, covariances and precisions
         of a sequence of single component of Gaussians
    beta: parameter for Gibbs prior
    adj_list: adjacency list where each element contains the neighour indices 
    ICM_max_iter: maximum number of iterations of the ICM algorithm
    rel_tol: relative tolerence for convergence
    
    Output:
    A dict with keys:
    label: MAP cluster labels
    U: final total energy
    convergence: 
        0: successfully converged; 
        1: iteration terminated due to the unique number of labels becomes
           less than the inital number of clusters during ICM iterations;
        2: exceed the maximum number of interations  
  
    Reference:
    [1] J.Besag, On the Statistical Analysis of Dirty Pictures, J. R. Statist. Soc. B (1986), 48, No. 3
    [2] Y.Zhang, M.Brady and S.Smith, Segmentation of Brain MR Images through a Hidden Markov Random Field Model and 
        the Expectation-Maximization Algorithm, IEEE Trans Medical Imaging (2001), Vol. 20 No. 1
    """
    N, P = X.shape
        
    G_stats = get_gaussian_stats(GMMobj)
    K = G_stats['n_components']
    means = G_stats['means']
    Covs = G_stats['covariances'] 
 
    sum_U_old = 0
    sum_U_new = 0
    
    Y = Y_init
    U1 = np.array([-1*multivariate_normal.logpdf(X,mean=means[k,:],cov=Covs[k,:,:],allow_singular=allow_singular) for k in xrange(K)])
    #U1 = np.array([-1*multinormal_pdf(X,mean=means[k,:],cov=Covs[k,:,:],log=True,pseudo_inv=allow_singular) for k in xrange(K)])
    U1 = U1.T    
    
    it = 1
    converged = False
    converge_flag = 0
    while it==1 or (it<=ICM_max_iter and not converged):                       
        U2 = clique_protential(Y, adj_list, beta)
        U = U1+U2
    
        Y_new = np.argmin(U,1)
        if len(np.unique(Y_new))<K:
            converge_flag = 1
            break
        else:
            Y = Y_new
            
        U_rowMin = np.min(U,1)
        sum_U_old = sum_U_new
        sum_U_new = np.sum(U_rowMin)
        
        converged = np.linalg.norm(sum_U_old-sum_U_new) <= rel_tol*np.linalg.norm(sum_U_old)
        it += 1
        if it > ICM_max_iter:
            converge_flag = 2
        
    return dict(label=Y, U=sum_U_new, convergence=converge_flag)


def GMM_HMRF_EM(X, beta, adj_list, GMM_init, EM_max_iter=100, ICM_max_iter=20, rel_tol=1e-8, soft=True, allow_singular=True):
    """
    The EM algorithm for unsupervised hidden MRFs. The process of maximizing pseudo-likelihood 
    is obtained by ICM algorithm.
    
    Input:
    X: data (n_samples, n_features)
    GMM_init: initial Gaussian mixture model object
    beta: parameter for Gibbs prior
    adj_list: adjacency list where each element contains the neighour indices 
    EM_max_iter: maximum number of iterations of the EM algorithm
    ICM_max_iter: maximum number of iterations of the MAP algorithm
    rel_tol: relative tolerence for convergence

    Output:
    label: final cluster labels
    GMM: a dict containing means, covariances and precisions
         of a sequence of single component of Gaussians 
    convergence: 
        0: successfully converged; 
        1: iteration terminated due to the unique number of labels becomes
           less than the inital number of clusters during ICM iterations;
        2: exceed the maximum number of ICM iterations 
        3: exceed the maximum number of EM iterations 
    reference:
    [1] J.Besag, On the Statistical Analysis of Dirty Pictures, J. R. Statist. Soc. B (1986), 48, No. 3
    [2] Y.Zhang, M.Brady and S.Smith, Segmentation of Brain MR Images through a Hidden Markov Random Field Model
        and the Expectation-Maximization Algorithm, IEEE Trans Medical Imaging (2001), Vol. 20 No. 1
    """
    N, P = X.shape    
    Y_init = GMM_init.fit(X).predict(X)
    GMMobj = GMM_init

    sum_U_new = 0 # the total potential value of all samples
    sum_U_old = 0    
    it = 1
    converged = False
    converge_flag = 0
    
    while (it==1) or (it<=EM_max_iter and not converged):
        sum_U_old = sum_U_new
        
        # E-step:
        # update cluster label Y and the corresponding posteriors
        MAP_Y = MRF_MAP_ICM(X, Y_init, GMMobj, beta, adj_list, ICM_max_iter, rel_tol)
        Y = MAP_Y['label']
        sum_U_new = MAP_Y['U']
        converge_flag = MAP_Y['convergence']
        if converge_flag==1:
            break    
        
        if soft:
            weights = label_posterior(X, Y, GMMobj, adj_list, beta)
        else:
            weights = label_indicator(X, Y)
            
        # M-step:
        # update GMM parameters             
        GMMobj = MLE_GMM_stats(X,weights,nonsingular_cov=allow_singular)
        
        # NOTE: Technically, convergance should be determined by the difference 
        # between GMM parameters of two consecutive interations. However, this is
        # roughly equivalent to comparing energies of distributions 
        # of two consecutive interations. 
        converged = np.linalg.norm(sum_U_old-sum_U_new) <= rel_tol*np.linalg.norm(sum_U_old)
        
        Y_init = Y
        it += 1
        if it > EM_max_iter:
            converge_flag = 3

    return dict(label=Y,GMM=GMMobj,convergence=converge_flag)


def rank_gmm_features(GMMobj,featureNames):
    """
    Return a list of length of number of clusters
    Each element of the returned list is a tuple which constains
    a. feature names
    b. mean values ranked by corresponding center coordinates (means) 
    c. covariance matrix with rows/columns ordered by ranked features
    """
    K = len(GMMobj['weights_'])
    centers = GMMobj['means_']  # shape(n_components, n_features)
    Covs = GMMobj['covariances_'] # shape(n_components, n_features, n_features)

    descend_sort = lambda x: np.sort(x)[::-1]
    descend_argsort = lambda x: np.argsort(x)[::-1]
    sorted_means = np.apply_along_axis(descend_sort, axis=1, arr=centers)
    sorted_idx = np.apply_along_axis(descend_argsort, axis=1, arr=centers)

    ranked_features = [featureNames[sorted_idx[i,:]] for i in xrange(K)]            
    ranked_features_inCluster = [(ranked_features[i].tolist(),sorted_means[i].tolist(),Covs[i,sorted_idx[i,:],:][:,sorted_idx[i,:]]) 
                                 for i in xrange(K)]                             
    return ranked_features_inCluster    
        
def save_cluster_info(GMMdict, Ncomponents=None, beta=0,ranked_features=None,file_path='',model_name=''):
    if 'convergence' in GMMdict.keys():
        GMMdict.pop('convergence')
    GMMdict['beta'] = beta
    GMMdict['n_components'] = Ncomponents
    GMMdict['ranked_features'] = ranked_features 
    savefile = file_path+'clusters_'+model_name+'_Ncomp_'+str(Ncomponents)+'_beta_'+str(beta).replace('.','_')+'.pkl'
    with open(savefile,'wb') as output:
        pickle.dump(GMMdict, output, pickle.HIGHEST_PROTOCOL)

    
if __name__=='__main__':
    import re
    import time
    import sys
    sys.path.append('..')
    from Misc.ComdArgParse import ParseArg
    
    args = ParseArg()
    infiles = args['input']
    outpaths = args['output']
    params = args['param']
   
    infile_match = re.match('([\w\./]+) ([\w\./]+) ([\w\./]+)',infiles)     
    grid_pkl, graph_pkl, feature_pkl = infile_match.group(1), infile_match.group(2), infile_match.group(3)
    
    outpath_match = re.match('([\w\./]+) ([\w\./]+)',outpaths)
    figpath, clusterpath = outpath_match.group(1), outpath_match.group(2)
    
    param_match = re.match('(\d+) ([-+]?\d*\.\d+|\d+) ([-+]?\d*\.\d+|\d+)', params)
    Ncomponents = int(param_match.group(1))
    beta = float(param_match.group(2)) #smoothing prior parameter
    gamma = float(param_match.group(3)) #weight_concentration_prior
                      
    # load grid info   
    with open(grid_pkl,'rb') as input_file:
        grid_list = pickle.load(input_file)
    _, grd_x, grd_y, _, mask_grdInCity, _ = grid_list  
    cellsize = (abs(grd_x[1]-grd_x[0]), abs(grd_y[1]-grd_y[0]))
    grid_2d = (grd_x, grd_y)
    Ngrids = np.nansum(mask_grdInCity) 
    
    # load adjacency list
    with open(graph_pkl,'rb') as input_file:
        _, _, adj_list = pickle.load(input_file)

    # load features   
    with open(feature_pkl,'rb') as input_file:
        data_df = pickle.load(input_file)
        
    featureNames = data_df.columns.values
    X = data_df.values        
        
    zscore_scaler = preprocessing.StandardScaler()
    X_s = zscore_scaler.fit_transform(X)
        
    r_seed = 1234
    cov_type = 'full'
    n_init = 1
    max_iter_GMM = 200
    max_iter_BGMM = 1000
    EM_max_iter = 100
    ICM_max_iter = 30
          
    # inital GMM clustering
    GMM_init = GaussianMixture(n_components=Ncomponents, covariance_type=cov_type, n_init=n_init,
                               init_params='kmeans', max_iter=max_iter_GMM, random_state=r_seed).fit(X_s) 

    y_init= GMM_init.predict(X_s)  
    fig = plt.figure(figsize=(10,9)); ax = fig.add_subplot(111) 
    vc.plot_clusters(ax, y_init, grid_2d, flattened=True, mask=mask_grdInCity)          
    vc.colorbar_index(ncolors=Ncomponents,ax=ax,shrink=0.6)  
    figname = 'GMM_segmentation.png'    
    vc.save_figure(figpath+figname,fig)
    
    # save GMM cluster info
    GMM_dict = dict(label=y_init,GMMobj=GMM_init)
    save_cluster_info(GMM_dict,Ncomponents,file_path=clusterpath,model_name='GMM')
    
    # inital variational Bayesian GMM clustering
    BGMM_init = BayesianGaussianMixture(n_components=Ncomponents, n_init=n_init, init_params='kmeans', 
                                        weight_concentration_prior_type="dirichlet_process", 
                                        weight_concentration_prior=gamma, covariance_type=cov_type,
                                        max_iter=max_iter_BGMM, random_state=r_seed).fit(X_s)
    y_init= BGMM_init.predict(X_s)  
    fig = plt.figure(figsize=(10,9)); ax = fig.add_subplot(111) 
    vc.plot_clusters(ax, y_init, grid_2d, flattened=True, mask=mask_grdInCity)          
    vc.colorbar_index(ncolors=Ncomponents,ax=ax,shrink=0.6) 
    figname = 'BGMM_segmentation.png'    
    vc.save_figure(figpath+figname,fig)
    
    # save BGMM cluster info
    BGMM_dict = dict(label=y_init,GMMobj=BGMM_init)
    save_cluster_info(BGMM_dict,Ncomponents,file_path=clusterpath,model_name='BGMM')
    
    # fit GMM_HMRF 
    start = time.time()
    mrf = GMM_HMRF_EM(X_s, beta, adj_list, BGMM_init, EM_max_iter=EM_max_iter, ICM_max_iter=ICM_max_iter, soft=True, allow_singular=False)
    end = time.time()
    print 'Elapsed time:', round(end-start,2)  #
    print('Convergence: %d' % mrf['convergence'])
    
    fig = plt.figure(figsize=(10,9)); ax = fig.add_subplot(111) 
    vc.plot_clusters(ax, mrf['label'], grid_2d, flattened=True, mask=mask_grdInCity)        
    vc.colorbar_index(ncolors=Ncomponents,ax=ax,shrink=0.6)  
    figname = 'HMRF_GMM_segmentation.png'    
    vc.save_figure(figpath+figname,fig)
    
    # ranked features in each cluster
    ranked_features = rank_gmm_features(mrf['GMM'],featureNames) 
    for i in xrange(Ncomponents):
        fig, axes = plt.subplots(1, 3, figsize=(18,7)) 
        fig.subplots_adjust(bottom=0.5, top=0.9, wspace=1.4)
        vc.plot_GMM_stats(axes,ranked_features[i],['mean','Cov','Corr'])
        fig.suptitle('Cluster '+str(i),fontsize=12)
        figname = 'HMRF_Cluster'+str(i)+'_stats.png'    
        vc.save_figure(figpath+figname,fig)
        
    # save MRF_GMM cluster info
    save_cluster_info(mrf,Ncomponents,beta,ranked_features,file_path=clusterpath,model_name='MRF_GMM')
                
    

       