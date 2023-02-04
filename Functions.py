# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 22:59:34 2023

@author: Sujeni
"""
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA


def degrees(G):
    
     
    W, W_pos, W_neg =weights(G)
    
    d_pos=np.sum(W_pos, axis=1)
    d_neg = np.sum(W_neg, axis=1)
    d=d_pos+ d_neg
    return d,  d_pos, d_neg


def weights(G):
    
    W=nx.adjacency_matrix(G).todense() #adjacency matrix
    W=np.asarray(W)

    W_pos = np.zeros_like(W)
    W_neg = np.zeros_like(W)
    

   
    #non negative weight matrices
    W_pos[W>0] =1
    W_neg[W<0] = 1
    
   
    return W, W_pos, W_neg


def spectral_clustering(L, N = 10):
    """
    L: Laplacian
    N: Number of clusters
    """
    #eigenvalues and eigenvvectors
    vals, vecs = np.linalg.eig(L)

    # sort these based on the eigenvalues
    vecs = vecs[:,np.argsort(vals)]
    vals = vals[np.argsort(vals)]

    # kmeans on first n vectors with nonzero eigenvalues
    i = np.nonzero(vals)[0][0] #first index with non-zero element
    
    kmeans = KMeans(n_clusters=N)
    kmeans.fit(vecs[:,i:i+N].real)
    colors = kmeans.labels_
    
    return colors


