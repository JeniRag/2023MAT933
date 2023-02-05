# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 17:19:58 2023

@author: Sujeni
"""
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA
import time

def triads(G): #returns number or Triangles
    tri = nx.triangles(G)
    return np.array(list(tri.values())).sum()//3

def triangle_list(G): #returns triangle combinations
    cli_3_list=[]
    for each_clique_all_size in nx.enumerate_all_cliques(G) :
        if len(each_clique_all_size) <3:
    
            continue
    
        elif len(each_clique_all_size) == 3:
    
    
            cli_3_list.append(sorted(each_clique_all_size))
        
        else:
            break   # this stops after size 3 is done
        
    return cli_3_list

def triad_types(G):
    triangles = triangle_list(G)#[c for c in nx.enumerate_all_cliques(G) if len(c)==3]  #[c for c in nx.cycle_basis(G) if len(c)==3]
    #edges=set(G.edges)
    n_unbalanced = 0
    n_strongly_balanced=0
    n_weakly_balanced = 0
    n_triangles = 0

    for triad in triangles:
        
        
        e1,e2,e3 = triad
        
        #edge_set = {(e1, e2), (e2, e3), (e3, e1)}

        #if edge_set.issubset(edges):
            #G.has_edge()
             #get_edge_data
        if (G.has_edge(e1, e2)==True) & (G.has_edge(e2, e3)==True) &( G.has_edge(e1, e3)==True):
            n_triangles += 1
            neg=0
            pos = 0

            w1=G.get_edge_data(e1, e2)["weight"]
            w2=G.get_edge_data(e2, e3)["weight"]
            w3=G.get_edge_data(e1, e3)["weight"]

            #print("{} , {} : {}".format(e1, e2, w1))
            #print("{} , {} : {}".format(e2, e1, w2))
            #print("{} , {} : {}".format(e1, e3, w3))
            
            
            for w in [w1, w2, w3]:
                if w<0:
                    neg +=1

                elif w>0:
                    pos +=1


            if (neg==1) & (pos==2):
                n_unbalanced += 1
            
            elif pos==3:
                n_strongly_balanced += 1
            
            elif (pos==1) & (neg==2):
                n_strongly_balanced += 1
                
            elif neg==3:
                n_weakly_balanced += 1
     
    return n_unbalanced, n_weakly_balanced, n_strongly_balanced, n_triangles


def CBI(G, group_labels): #Community balance index 
    """
    G: network
    group_labels: 1D array with corresponding cluster element for each node i
    
    """
    """
    n_c: community size
    k: total number of communities in clustering
    N_PPNC: number of unbalance triads in community c 
    N_triadsc: total number of triads in community c
    greater CBI -> better clustering
    
    """
    nodes=list(G.nodes())
    nodes=np.asarray(nodes)
    values, counts = np.unique(group_labels, return_counts=True)
    
    nenner = 0
    n_wb =0
    n_sb=0
    n_ub=0
    n_tri = 0
    
    for i,c in enumerate(values):
        n_c = counts[i] #n_c
        cj=(group_labels==c) #selected group
        
        # sub_nodes=[]
        
        # for i in range(len(group_labels)):
        #     if cj[i]: #if node belongs to the selected group, add to sub node
        #         sub_nodes.append(nodes[i])
        
        sub_nodes=nodes[cj]
    
        #print("class {} subnodes:{}".format(c, sub_nodes))
        Gt_sub = G.subgraph(sub_nodes)
        #PPN_c = unbalanced_tri(Gt_sub)
        PPN_c, n_wb_c, n_sb_c, n_triangles_c= triad_types(Gt_sub)
        
        n_ub += PPN_c
        n_wb += n_wb_c
        n_sb += n_sb_c
        n_tri += n_triangles_c
        #print("unbalanced triangles of class {}: {}".format(c,PPN_c))

        triads_c= triads(Gt_sub) #total number of triads
        if triads_c==0:
            continue#then nenner +=0
        else:
            nenner += (n_c* (1- PPN_c/triads_c))
    
    
    print("unbalanced triangels: {} , {}%".format(n_ub, n_ub/n_tri*100))
    print("weakly balanced: {}, {}%".format(n_wb, n_wb/n_tri*100))
    print("strongly balanced: {}, {}%".format(n_sb, n_sb/n_tri*100))
    print("total triangles: {}".format(n_tri))
    return nenner/counts.sum() *100