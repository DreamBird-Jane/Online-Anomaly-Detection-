#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:41:06 2019

@author: jane_hsieh

Goal: 
    1. Use method of "frequent directions"
        to compute sketch matrix B with dim=(l,d) from pd. matrix A with dim=(n,d))

Source paper:
    1. Efficient anomaly detection via matrix sketching
        Read more: http://papers.nips.cc/paper/8030-efficient-anomaly-detection-via-matrix-sketching
        Note: for application of "frequent directions"
    2. Frequent Directions: Simple and Deterministic Matrix Sketching
        Read More: https://epubs.siam.org/doi/abs/10.1137/15M1009718?
        Note: Specify the algorithms of "frequent directions"

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#================= Input original data A and output sketch B ========================================================


#---- Define functions: Frequent directions(path, l, n, d)-> B (sketch)-------------------------
def Frequent_directions(path, l, n, d):
    '''
    Note:
        1. refer to Algorithm 1 in paper:
            "Frequent Directions: Simple and Deterministic Matrix Sketching"
        2. The data matrix A where 'path' locate should be a dataframe with
            (1) 1st column = index
            (2) 1nd row = column names
    
    Input:
        path: string of directory path to data A position; 
        l: sketch size, to reduce n rows of A into l frequent rows; usually l <= d 
            (since there're at most d row vectors to span a d-dim row space)
        n, d: the dimentsion of original data matrix A s.t. dim(A)=(n,d)
            n for instances/observations, d for dim. of measurement each time
    Output:
        B: sketch of data matric A; dim(B)=(l,d)
            If l <= d, the B=U*diag(s)*Vt with dim(U)=(l,l), dim(s)= l, dim(Vt) = (l,d)
    '''

    B = np.zeros((l,d))
    
    a_reader = pd.read_csv(path, index_col=0, chunksize=1)  #read the data row by row
    for a in a_reader:
        print(a.index)
        B[l-1] = a
        U, s, Vt = np.linalg.svd(B)
        Vt = Vt[:l,:]
        
        #temp = Sigma**2
        temp = np.diag(s)**2
        delta = temp[l-1,l-1]        
        temp2 = np.sqrt(  temp - np.diag(np.repeat(delta,l))  )
        
        #update B
        B = np.dot(temp2,Vt)
        
    return B
        
#-------------------------------------------------------------------------------------------------



#============================ Calculation of sketches B ======================================
