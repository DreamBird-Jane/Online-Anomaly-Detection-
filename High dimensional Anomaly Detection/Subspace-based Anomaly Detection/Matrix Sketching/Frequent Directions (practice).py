#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 12:55:00 2019

@author: jane_hsieh

Goal: 
    1. Use method of "frequent directions"
        to compute sketch matrix B with dim=(l,d) from data matrix A with dim=(n,d))

Source paper:
    1. Efficient anomaly detection via matrix sketching
        Read more: http://papers.nips.cc/paper/8030-efficient-anomaly-detection-via-matrix-sketching
        Note: for application of "frequent directions"
    2. Frequent Directions: Simple and Deterministic Matrix Sketching
        Read More: https://epubs.siam.org/doi/abs/10.1137/15M1009718?
        Note: Specify the algorithms of "frequent directions"
Data:
dataXpart_c--
    Note: Xpart of K8.data (#{columns}=5409; #{rows}=16772) with row of missing data dropped
    #{columns}=5408; #{rows}=16592 (i.e.1 column of Y part dropped; 180 rows dropped)
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time



data_dir = '/Users/jane_hsieh/OneDrive - nctu.edu.tw/Data Science Analysis Templates/Datasets/p53 Mutantys Data Set/p53_old_2010'
work_dir = '/Users/jane_hsieh/OneDrive - nctu.edu.tw/Data Science Analysis Templates/Anomaly Detection/Matrix Sketching'
os.chdir(work_dir)
os.getcwd()


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

#---------------------------- Input paramters: path, l, n, d -------------------------------------------
#k: the number of right eigenvectors(V) extracted (A=U*diag(s)*Vt)
#Given k, the Theom 1 from Source paper 1. that l~k^2, but its experiment indicate that l~C*k (e.g. C=10) would be enough
#path: the path to data
path = data_dir+'/K8_Xpart_dropna_c.data'

n = 16592
d = 5408


#---------------------- Derive B given parameters ----------------------------------------------
#l =[100, 200, 400]   #Given k= 10, 20,30 

## Case 1: l=100
start = time.time() 

B = Frequent_directions(path, 100, n, d)  #or Frequent_directions(path=path, l=100, n=n, d=d) but not mixed

end = time.time()
print(end - start,'sec. \n') #unit: sec.
print('I.e., the running time is {:.3f} min.'.format((end - start)/60))    

#pd.DataFrame(B).to_csv(data_dir+'/K8_Xpart2_dropna_c_B.data', header=None, index=None)

'''
Note: B didn't run through all rows of dataXpart_c but only to the row_index= 5766; to this far,
B is considered the sketch matrix of "dataXpart5766_c"
'''
pd.DataFrame(B).to_csv(data_dir+'/K8_Xpart2_dropna_c_5766_B.data', header=None, index=None)



dataXpart_c = pd.read_csv(data_dir+'/K8_Xpart_dropna_c.data', index_col=0)
dataXpart5766_c= dataXpart_c.loc[1:5766]
dataXpart5766_c.to_csv(data_dir+'/K8_Xpart_dropna_c_5766.data')


