#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
==================================
Gragh
==================================
Created on Fri Oct  7 16:56:25 2016

@author: xiaomuliu
"""

import numpy as np
import networkx as nx

def pair_node_idx(node_idx_array):
    """
    Convert node index array of form: tuple(np.array[x1,x2,...], np.array[y1,y2,...])
    to the form of [(x1,y1),(x2,y2),...]
    """
    xy_list = np.c_[node_idx_array[0], node_idx_array[1]]
    paired_node_list = [(xy[0],xy[1]) for xy in xy_list]
                         
    return paired_node_list


def arr_idx_to_flat_idx(node_list, M, N, order='C'):
    """
    Convert keys and values in node list from array index to flattened index
    M, N are the dimensions corresponds to array index
    Use numpy.unravel_index and numpy.ravel_multi_index to convert between
    index arrays and an array of flat indices
    """
    node_list_ravel = {}
    for node, adjnodes in node_list.items():
        key = np.ravel_multi_index(node,(M,N), order=order)
        val = [np.ravel_multi_index(nb,(M,N), order=order) for nb in adjnodes]
        node_list_ravel[key] = val  
    
    return node_list_ravel

def flat_idx_to_mask_idx(adj_list, mask_idx):
    """
    Convert keys and values (flattened indices from an array) in adjacency list 
    to the corresponding index of masked indices
    """
    adj_list_mask = {}
    for node, adjnodes in adj_list.items():
        key = np.where(mask_idx==node)[0].squeeze().tolist()
        val = [np.where(mask_idx==nb)[0].squeeze().tolist() for nb in adjnodes]
        adj_list_mask[key] = val  
    
    return adj_list_mask    
    
def grd_to_adj_list(M, N, mask=None, order='C'):
    """
    Return the adjacency list given the grid of shape (M,N) (and the mask).
    """
    if mask is None:
        mask = np.ones(M*N).astype('bool')
    
    graph = nx.grid_2d_graph(M,N)     
        
    node_inMask_ravel = np.where(mask)[0]  # the flattened indices of mask
    node_inMask_im = np.unravel_index(node_inMask_ravel, (M,N), order=order) 
    paired_node_list = pair_node_idx(node_inMask_im)
    
    adj_idx_arr_list = nx.to_dict_of_lists(graph,nodelist=paired_node_list)
    adj_idx_ravel_list = arr_idx_to_flat_idx(adj_idx_arr_list, M, N, order=order)
    
    adj_mask_idx_ravel_list =  flat_idx_to_mask_idx(adj_idx_ravel_list, node_inMask_ravel) 
    
    return adj_idx_arr_list, adj_idx_ravel_list, adj_mask_idx_ravel_list
   
    
if __name__=='__main__':
    import cPickle as pickle
    import sys
    sys.path.append('..')
    from Misc.ComdArgParse import ParseArg
    
    args = ParseArg()
    infile = args['input']
    outpath = args['output']
   
    grid_pkl = infile if infile is not None else "../SharedData/SpatialData/grid.pkl"
    filePath_save = outpath if outpath is not None else "../SharedData/SpatialData/" 

    # load grid info  
    with open(grid_pkl,'rb') as input_file:
        grid_list = pickle.load(input_file)
    _, grd_x, grd_y, _, mask_grdInCity, _ = grid_list  
    cellsize = (abs(grd_x[1]-grd_x[0]), abs(grd_y[1]-grd_y[0]))  
    grid_2d = (grd_x, grd_y)
    Nx, Ny = len(grd_x),len(grd_y)
    Ngrids_inCity = np.nansum(mask_grdInCity) 
            
    adj_idx_arr_list, adj_idx_ravel_list, adj_mask_idx_ravel_list = grd_to_adj_list(Nx, Ny, mask_grdInCity, order='F') 
    
    # save graph objects
    graph_pkl = filePath_save+'graph.pkl'
    graph_list = [adj_idx_arr_list, adj_idx_ravel_list, adj_mask_idx_ravel_list]
    with open(graph_pkl,'wb') as output:
        pickle.dump(graph_list, output, pickle.HIGHEST_PROTOCOL)