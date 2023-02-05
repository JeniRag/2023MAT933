# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 15:22:34 2023

@author: Sujeni
"""
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA

def build_L_BR(W_pos, W_neg):
    
    d_pos = np.sum(W_pos, axis=1)
    D_pos = np.diag(d_pos)
    
    L_BR = D_pos - W_pos + W_neg
    return L_BR

def build_L_BN(W_pos, W_neg):
    d = np.sum(W_pos+W_neg, axis=1)
    
    #dinv=np.zeros_like(d)
    
    D=np.diag(d)
    DInv = np.linalg.inv(D)
    
    L_BR = build_L_BR(W_pos, W_neg)
    
    L_BN = DInv@L_BR
    
    return L_BN

def build_L_SR(W_pos, W_neg):
    d = np.sum(W_pos + W_neg, axis=1)
    
    D=np.diag(d)
    L_SR = D-W_pos + W_neg
    return L_SR

def build_L_SN(W_pos, W_neg):
    d = np.sum(W_pos + W_neg, axis=1)
    
    
    dinv=np.zeros_like(d, dtype = float)
    
    dinv[d!=0] = 1./np.sqrt(d[d!=0])
    
    DInv=np.diag(dinv)
    
    L_SR =build_L_SR(W_pos, W_neg)
    L_SN = DInv @ L_SR @ DInv
    
    return L_SN

def build_L_AM(W_pos, W_neg):
    L_plus_sym=build_laplacian_matrix(W_pos)
    Q_minus_sym=build_signless_Laplacian(W_neg)
    L_AM=L_plus_sym + Q_minus_sym
    
    return L_AM
    
def build_laplacian_matrix(W):
    d=np.sum(W, axis=1)
    dInv=np.zeros_like(d, dtype=float)
    dInv[d>0] = 1./d[d>0]
   
    
    dInv = dInv**0.5
    DInv = np.diag(dInv)
    
    L=np.diag(d) -W
    
    #symmetric Laplacian
    L_sym=DInv@L@DInv
    #enforce symmetry
    #L = (L + L')/2;

    
    return L_sym

def build_signless_Laplacian(W):
    d=np.sum(W,axis=1)
    dInv=np.zeros_like(d, dtype = float)
    dInv[d>0] = 1./np.sqrt(d[d>0])
    
    dInv=dInv**0.5
    DInv=np.diag(dInv)
    
    Q = np.diag(d)+W
    Q_sym=DInv@Q@DInv
    
    #enforce symmetry
    #L = (L + L')/2;

    
    return Q_sym


def GM(A,B):
    #C=A@(np.linalg.inv(A)@B)**(1/2)
    assert np.all(np.linalg.eigvals(A) > 0) #positive definite
    assert np.all(np.linalg.eigvals(B) > 0)
    
   
    A=A+0j
    A_sqrt=np.sqrt(A+0j)
    non_zero=(A != 0.0)
    A_negsqrt=np.zeros_like(A,dtype=float)
    A_negsqrt[non_zero]= 1/np.sqrt(A[non_zero])
    
    m=A_negsqrt @ B @ A_negsqrt
    C=A_sqrt @ np.sqrt(m) @ A_sqrt
    
    
    #C=(A@ np.linalg.inv(B))**0.5@B
    
    return np.real(C)
    
   


def build_L_GM(W_pos, W_neg, shift = 1.e-6):
    N=W_pos.shape[0]
    I=np.eye(N)
    
    L_pos_sym=build_laplacian_matrix(W_pos)+I*shift
    Q_minus_sym=build_signless_Laplacian(W_neg) + I*shift
    
    L_GM = GM(L_pos_sym, Q_minus_sym)
    
    return L_GM