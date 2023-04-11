#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 19:30:03 2019

@author: jane_hsieh
Goal: 
    1. compute rank k leverage scores 
    2. compute rank k projection distances

Source paper:
    1. Efficient anomaly detection via matrix sketching
        Read more: http://papers.nips.cc/paper/8030-efficient-anomaly-detection-via-matrix-sketching
    [2. Frequent Directions: Simple and Deterministic Matrix Sketching
        Read More: https://epubs.siam.org/doi/abs/10.1137/15M1009718?
        ]
Data:
1.Original data matrix
    K8_Xpart_dropna_c_5766.data--
    Note: Xpart of K8.data (#{columns}=5409; #{rows}=16772) with row of missing data dropped
    #{columns}=5408; #{rows}=5681 (i.e.1 column of Y part dropped; 180 rows dropped)


Running time:
    1. for svd--  I.e., the running time is 1.570 min.
    2. for AD score calculation (k=7 only)-- the running time is 16.252 min.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time



data_dir = '/Users/jane_hsieh/OneDrive - nctu.edu.tw/Data Science Analysis Templates/Datasets/p53 Mutants Data Set/p53_old_2010'
work_dir = '/Users/jane_hsieh/OneDrive - nctu.edu.tw/Data Science Analysis Templates/Anomaly Detection/Subspace-based Anomaly Detection'
os.chdir(work_dir)
os.getcwd()

#================= Input data: origin (first time) ========================================================
# First time input of data

data = pd.read_csv(data_dir+'/K8_Xpart_dropna_c_5766.data', index_col=0)  #'header =0': no column names


data.dtypes
'''
    from data.dtypes, we observe that all variables have data type of 'float64', 
'''
data.isnull().sum().sum()
'''
    Since there're at least 57 missing values(NaN) for each column(variable), 
    we then drop rows whichever has any missing value.
'''
#Mean=data.mean()
#data_c = data.subtract(Mean)
'''
    Since mean of data is not 0, we subtract data into data_c to mean-center data.
    This is for the later SVD
'''


#======================== Simple EDA: Inspect data ====================================================

#-------------Full svd for data matrix: data ------------------------------------------
start1 = time.time() 

U, s, Vt = np.linalg.svd(data)

end1 = time.time()
print(end1 - start1,'sec. \n') #unit: sec.
print('I.e., the running time is {:.3f} min.'.format((end1 - start1)/60))    

'''
## check orthogonality
temp = np.dot(U.T,U)
temp = np.dot(Vt,Vt.T)
sns.heatmap(temp)
plt.show()
'''

#pd.DataFrame(U).to_csv(data_dir+'/K8_Xpart_dropna_c_6766_U.data', header=None, index=None)
pd.DataFrame(s).to_csv(data_dir+'/K8_Xpart_dropna_c_5766_s.data', header=None, index=None)
pd.DataFrame(Vt).to_csv(data_dir+'/K8_Xpart_dropna_c_5766_Vt.data', header=None, index=None)


#--------------------------------------------------------------------------------------------
#Investigate the number of right eigenvectors-- Vt.T (V) needed extracting
s_sq = s**2
temp = np.cumsum(s_sq);print(temp[0])
temp = temp/temp[-1];


##plot "Cumulated Variance explained"
fig = plt.figure()
plt.plot(temp*100)
plt.xlabel('k: # of eigenvectors (V)')
plt.ylabel('unit: %')
plt.title('Cumulated Variance explained')
plt.show()

# Specify the axes
plt.axes([0.4,0.3,0.4,0.4]) #(all in units relative to the figure dimensions)
# Plot the sliced series in red using the current axes
plt.plot(range(15), temp[:15]*100) #,color='blue'
plt.title('turing point')
plt.xticks(list(range(0,15,1)))
for x in range(3,8,2):
    plt.axvline(x=x, color='red', linestyle='--')
plt.axhline(y=70, color='green', linestyle='--')
plt.show()

fig.savefig(data_dir+'/K8_Xpart_dropna_c_5766_CVE.svg')
'''
    According to the plot of "Cumulated Variance explained", we choose k=3 (60.8%), 5(67.4%), 7(70.4%)**

'''


#===============================================================================================================
#========== Calculate Rank-k leverage scores/projection distance for each of of data matrix:data_c ============================
import PCAbased_AD_Algorithms as PCA_AD


start2 = time.time() 

AD_scores = {}

for k in [7]:
    print(k)
    name = 'L_'+str(k);print(name)
    AD_scores[name] = data.apply(PCA_AD.Rank_k_leverage_score,axis=1,args=[k,Vt,s])
    name = 'T_'+str(k);print(name)
    AD_scores[name] = data.apply(PCA_AD.Rank_k_projection_distance,axis=1,args=[k,Vt])

end2 = time.time()
print(end2 - start2,'sec. \n') #unit: sec.
print('I.e., the running time is {:.3f} min.'.format((end2 - start2)/60))    


AD_scores = pd.DataFrame(AD_scores, index = data.index)
AD_scores.to_csv(data_dir+'/K8_Xpart_dropna_c_5766_ADscores.data')










