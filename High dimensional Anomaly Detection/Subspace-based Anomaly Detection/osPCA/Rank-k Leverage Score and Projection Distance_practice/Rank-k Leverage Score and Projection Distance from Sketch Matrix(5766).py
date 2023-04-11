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
    #{columns}=5408; #{rows}=5681 
2. Sketch matrix B
    K8_Xpart_dropna_c_5766_B.data
    #{columns}=5408; #{rows}=100 

Running time:
    1. for svd--  I.e., the running time is 0.076 min.
    2. for AD score calculation (k=7 only)-- the running time is 0.150 min.

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
B = pd.read_csv(data_dir+'/K8_Xpart_dropna_c_5766_B.data', header=None, index_col=False, 
                   names=['v'+ str(i) for i in range(5408)])  #'header =0: no column names
#data is for later computation of AD scores
data = pd.read_csv(data_dir+'/K8_Xpart_dropna_c_5766.data', index_col=0)  #'header =0': no column names


B.dtypes
'''
    from data.dtypes, we observe that all variables have data type of 'float64', 
'''
B.isnull().sum().sum()
'''
    Since there're at least 57 missing values(NaN) for each column(variable), 
    we then drop rows whichever has any missing value.
'''
#Mean=B.mean()
#B_c = B.subtract(Mean)
'''
    Since mean of B is not 0, we subtract B into B_c to mean-center B.
    This is for the later SVD
'''


#======================== Simple EDA: Inspect data ====================================================

#-------------Full svd for data matrix: B_c ------------------------------------------
start1 = time.time() 

U, s, Vt = np.linalg.svd(B)

end1 = time.time()
print(end1 - start1,'sec. \n') #unit: sec.
print('I.e., the running time is {:.3f} min.'.format((end1 - start1)/60))    

Vt = Vt[:B.shape[0]]
pd.DataFrame(Vt).isnull().sum().sum()
'''
## check orthogonality
temp = np.dot(U.T,U)
temp = np.dot(Vt,Vt.T)
sns.heatmap(temp)
plt.show()
'''

#pd.DataFrame(U).to_csv(data_dir+'/K8_Xpart_dropna_c_6766_U.data', header=None, index=None)
#pd.DataFrame(s).to_csv(data_dir+'/K8_Xpart_dropna_c_5766_s.data', header=None, index=None)
#pd.DataFrame(Vt).to_csv(data_dir+'/K8_Xpart_dropna_c_5766_Vt.data', header=None, index=None)


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

fig.savefig(data_dir+'/K8_Xpart_dropna_c_5766_B_CVE.svg')
'''
    According to the plot of "Cumulated Variance explained", we choose k=10, 20,30
    which is also the numbers of k used in experiments in Resource paper 1.
'''


#==========================================================================================================================
#========== Calculate Rank-k leverage scores/projection distance for each of of data matrix:B_c ============================
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
AD_scores.to_csv(data_dir+'/K8_Xpart_dropna_c_5766_B_ADscores.data')




#==================== Comparison of AD scores: ================================================================================
#============== K8_Xpart_dropna_c_5766_ADscores.data vs K8_Xpart_dropna_c_5766_B_ADscores.data ================================
AD_scores0 = pd.read_csv(data_dir+'/K8_Xpart_dropna_c_5766_ADscores.data', index_col=0)
Vt0 = pd.read_csv(data_dir+'/K8_Xpart_dropna_c_5766_Vt.data', header=None, index_col=False, 
                   names=['v'+ str(i) for i in range(5408)]) 

#---------------------------------------------------------------------------------
#Check the Scatter plots of AD scores (L7, T7) between Data matrix A and sketch B
fig1 = plt.figure()
plt.scatter(AD_scores0.L_7, AD_scores.L_7,s=0.8)
plt.xlabel('Original Data matrix A')
plt.ylabel('Sketch matrix B')
plt.title('Rank k={} Leverage scores from A and B'.format(k))
plt.show()
fig1.savefig(data_dir+'/K8_Xpart_dropna_c_5766_Scatter_of_L7_with_B.svg')


fig2 = plt.figure()
plt.scatter(AD_scores0.T_7, AD_scores.T_7, s=0.8)
plt.xlabel('Original Data matrix A')
plt.ylabel('Sketch matrix B')
plt.title('Rank k={} Projection disctance from A and B'.format(k))
plt.show()
fig2.savefig(data_dir+'/K8_Xpart_dropna_c_5766_Scatter_of_T7_with_B.svg')
'''
    From the scatter plots,it seems that AD scores from A and B are not so comparible; they're
    not in linear relationship
'''

#---------------------------------------------------------------------------------
#Check the similarity of eigenvectors between Data matrix A and sketch B
temp = np.dot(Vt,Vt0[:B.shape[0]].T)

fig1=sns.heatmap(temp)
plt.xlabel('Original Data matrix A (first 100 eigenvectors)')
plt.ylabel('Sketch matrix B')
plt.xticks(rotation=90)
plt.title('Inner product(similarity) of eigenvectors between A and B')
fig1.figure.savefig(data_dir+'/K8_Xpart_dropna_c_5766_Similarity_of_Vt_with_B.svg')

##zoom in
fig2=sns.heatmap(temp[:10,:10])
plt.xlabel('Original Data matrix A (first 100 eigenvectors)')
plt.ylabel('Sketch matrix B')
plt.xticks(rotation=90)
plt.title('Inner product(similarity) of eigenvectors between A and B')
fig2.figure.savefig(data_dir+'/K8_Xpart_dropna_c_5766_Similarity_of_Vt_with_B(2).svg')
